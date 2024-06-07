const tf = require('@tensorflow/tfjs');

// Exemple de données textuelles et de labels
const sentences = [
  "I love machine learning", // Positive
  "TensorFlow is great for deep learning", // Positive
  "Natural language processing is a complex field", // Positive
  "I hate bugs in the code", // Negative
  "Debugging is a tedious task", // Negative
  "I dislike syntax errors" // Negative
];

const labels = [1, 1, 1, 0, 0, 0]; // 1: Positive, 0: Negative

// Fonction de tokenisation
class Tokenizer {
  constructor() {
    this.wordIndex = {};
  }

  fitOnTexts(texts) {
    let index = 1;
    texts.forEach(text => {
      text.split(' ').forEach(word => {
        if (!this.wordIndex[word]) {
          this.wordIndex[word] = index++;
        }
      });
    });
  }

  textsToSequences(texts) {
    return texts.map(text =>
      text.split(' ').map(word => this.wordIndex[word])
    );
  }
}

// Prétraitement des données textuelles
const tokenizer = new Tokenizer();
tokenizer.fitOnTexts(sentences);

const sequences = tokenizer.textsToSequences(sentences);
const maxLen = Math.max(...sequences.map(seq => seq.length));
const paddedSequences = sequences.map(seq => {
  const padding = Array(maxLen - seq.length).fill(0);
  return padding.concat(seq);
});

const inputTensor = tf.tensor2d(paddedSequences, [paddedSequences.length, maxLen]);
const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

// Définir le modèle
const model = tf.sequential();
model.add(tf.layers.embedding({
  inputDim: Object.keys(tokenizer.wordIndex).length + 1,
  outputDim: 50,
  inputLength: maxLen
}));
model.add(tf.layers.lstm({units: 100}));
model.add(tf.layers.dense({units: 50, activation: 'relu'}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));

model.compile({
  optimizer: 'adam',
  loss: 'binaryCrossentropy',
  metrics: ['accuracy']
});

// Entraîner le modèle
model.fit(inputTensor, labelTensor, {
  epochs: 10
}).then(() => {
  console.log('Model trained');
  
  // Extraire les représentations des phrases
  const intermediateModel = tf.model({inputs: model.input, outputs: model.layers[1].output});
  const embeddings = intermediateModel.predict(inputTensor);
  embeddings.array().then(array => {
    console.log('Embeddings:', array);
  });
});
