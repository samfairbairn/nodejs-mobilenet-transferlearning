// @ts-check
const tf = require("@tensorflow/tfjs-node");

const minimist = require("minimist");
const model = require("./model");
const data = require("./data");
const ui = require("./ui_mock");

const Model = new model();

let args = minimist(process.argv.slice(2), {
    string: ["images_dir", "model_dir"],
    boolean: true,
    default: {
        skip_training: false,
        batch_size_fraction: 0.2,
        dense_units: 100,
        epochs: 50,
        learning_rate: 0.0001
    }
});

// if (!args.images_dir) {
//     throw new Error("--images_dir not specified.");
// }

if (!args.model_dir) {
    throw new Error("--model_dir not specified.");
}

async function init() {
    if (!args.imageSrc) {
      await data.loadLabelsAndImages(args.images_dir).catch((e) => { console.log('data.loadLabelsAndImages:', e) });
    }

    console.time("Loading Model");
    await Model.init().catch((e) => { console.log('Model.init:', e) });
    console.timeEnd("Loading Model");
}

async function testModelSingleItem(imgSrc) {
  console.log("Testing Model: testModelSingleItem", imgSrc);
  await Model.loadModel(args.model_dir).catch((e) => { console.log('testModel: loadModel:', e) });

  if (Model.model) {
      console.time("Testing Predictions");
      console.log(Model.model.summary());

      let embeddings = await data.fileToTensor(imgSrc)
      console.log('embeddings', embeddings)

      tf.tidy(() => {
          // let embeddings = data.getEmbeddingsForImage(0)
          let prediction = Model.getPrediction(embeddings)
          console.log('prediction', prediction)
          // return prediction
      })
  }
}

async function testModel() {
    console.log("Testing Model");
    await Model.loadModel(args.model_dir).catch((e) => { console.log('Model.loadModel:', e) });

    if (Model.model) {
        console.time("Testing Predictions");
        console.log(Model.model.summary());

        let totalMislabeled = 0;
        let mislabeled = [];
        let imageIndex = 0;
        data.labelsAndImages.forEach(item => {
            let results = [];
            item.images.forEach(img_filename => {
                tf.tidy(() => {
                    let embeddings = data.dataset
                        ? data.getEmbeddingsForImage(imageIndex++)
                        : data.fileToTensor(img_filename);

                    let prediction = Model.getPrediction(embeddings);
                    results.push({
                        class: prediction.label,
                        probability: (
                            Number(prediction.confidence) * 100
                        ).toFixed(1)
                    });
                    if (prediction.label !== item.label) {
                        mislabeled.push({
                            class: item.label,
                            prediction: prediction.label,
                            filename: img_filename
                        });
                        totalMislabeled++;
                    }
                });
            });
            console.log({
                label: item.label,
                predictions: results.slice(0, 10)
            });
        });
        console.timeEnd("Testing Predictions");
        console.log(mislabeled);
        const totalImages = data.labelsAndImages
            .map(item => item.images.length)
            .reduce((p, c) => p + c);
        console.log(`Total Mislabeled: ${totalMislabeled} / ${totalImages}`);
    }
}

async function trainModel() {
    if (data.dataset.images) {
        const trainingParams = {
            batchSizeFraction: args.batch_size_fraction,
            denseUnits: args.dense_units,
            epochs: args.epochs,
            learningRate: args.learning_rate,
            trainStatus: ui.trainStatus
        };

        const labels = data.labelsAndImages.map(element => element.label);
        const trainResult = await Model.train(
            data.dataset,
            labels,
            trainingParams
        ).catch((e) => { console.log('Model.train:', e) });
        console.log("Training Complete!");
        const losses = trainResult.history.loss;
        console.log(
            `Final Loss: ${Number(losses[losses.length - 1]).toFixed(5)}`
        );

        console.log(Model.model.summary());
    } else {
        new Error("Must load data before training the model.");
    }
}

init()
    .then(async () => {
        if (!args.imageSrc) {
          await data.loadTrainingData(Model.decapitatedMobilenet).catch((e) => { console.log('data.loadTrainingData:', e) });
          console.log("Loaded Training Data");
        }

        if (args.skip_training) return;

        try {
            await trainModel();

            await Model.saveModel(args.model_dir);
        } catch (error) {
            console.error(error);
        }
    })
    .then(() => {
      if (args.imageSrc) {
        testModelSingleItem(args.imageSrc)
      } else {
        testModel();
      }
    });
