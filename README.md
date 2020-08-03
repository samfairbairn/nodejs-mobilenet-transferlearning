# nodejs-mobilenet-transferlearning

Retraining mobilenet (v1) image classification model using TensorFlow.js in Node.

Clone of [https://github.com/adwellj/node-tfjs-retrain](https://github.com/adwellj/node-tfjs-retrain) with some minor minor changes.

## Save Model Locally

Save a model locally matching the model defined in [model.js](model.js#10)

Save JSON file [mobilenet_v1_0.25_224/model.json](https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json)

Save weights

```
cat model.json | jq -r ".weightsManifest[].paths[0]" | sed 's/^/https:\/\/storage.googleapis.com\/tfjs-models\/tfjs\/mobilenet_v1_0.25_224\//' |  parallel curl -O
```

## Example Usages

-   Retrain and test model:  
    `node app.js --images_dir="dataset" --model_dir="mobilenet_v1_0.25_224"`
-   Skip retraining; just test model:  
    `node app.js --images_dir="dataset" --model_dir="mobilenet_v1_0.25_224" --skip_training=true`
-   Test a single image:  
    `node app.js --model_dir="mobilenet_v1_0.25_224" --imageSrc="test/image/location.png"`
-   Create sample images:  
    `node create_images.js C:/Retraining_Project/Images`

