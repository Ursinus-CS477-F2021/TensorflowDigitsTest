/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */


// Hyperparameters.
const BATCH_SIZE = 512;

// Data constants.
const IMAGE_RES = 28;

function getModel() {
  const model = tf.sequential();
  
  // In the first layer of our convolutional neural network we have 
  // to specify the input shape. Then we specify some parameters for 
  // the convolution operation that takes place in this layer.
  model.add(tf.layers.conv2d({
    inputShape: [IMAGE_RES, IMAGE_RES, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));

  // The MaxPooling layer acts as a sort of downsampling using max values
  // in a region instead of averaging.  
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Repeat another conv2d + maxPooling stack. 
  // Note that we have more filters in the convolution.
  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'varianceScaling'
  }));
  model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));
  
  // Now we flatten the output from the 2D filters into a 1D vector to prepare
  // it for input into our last layer. This is common practice when feeding
  // higher dimensional data to a final classification output layer.
  model.add(tf.layers.flatten());

  // Our last layer is a dense layer which has 10 output units, one for each
  // output class (i.e. 0, 1, 2, 3, 4, 5, 6, 7, 8, 9).
  const NUM_OUTPUT_CLASSES = 10;
  model.add(tf.layers.dense({
    units: NUM_OUTPUT_CLASSES,
    kernelInitializer: 'varianceScaling',
    activation: 'softmax'
  }));

  
  // Choose an optimizer, loss function and accuracy metric,
  // then compile and return the model
  const optimizer = tf.train.adam();
  model.compile({
    optimizer: optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  return model;
}


let startTime = 0;
let globalModel = null;
let epoch = 0;
const onBatchEnd = (batch, logs) => {
  statusElement.innerHTML = "Finished epoch " + epoch + " batch " + batch +
    "<BR>acc = " + logs.acc + "<BR>loss = " + logs.loss + 
    "<BR>Elapsed Time: " + Math.round((performance.now()-startTime)/1000);
}
const onEpochEnd = e => {
  epoch = e;
}


// Train the model.
async function modelTrain(data) {
  const BATCH_SIZE = 512;
  
  const [trainXs, trainYs] = tf.tidy(() => {
    const d = data.nextTrainBatch(NUM_TRAIN_ELEMENTS);
    return [
      d.xs.reshape([NUM_TRAIN_ELEMENTS, 28, 28, 1]),
      d.labels
    ];
  });
  
  const [testXs, testYs] = tf.tidy(() => {
    const d = data.nextTestBatch(NUM_TEST_ELEMENTS);
    return [
      d.xs.reshape([NUM_TEST_ELEMENTS, 28, 28, 1]),
      d.labels
    ];
  });

  globalModel = getModel();
  startTime = performance.now();
  globalModel.fit(trainXs, trainYs, {
    batchSize: BATCH_SIZE,
    validationData: [testXs, testYs],
    epochs: 10,
    shuffle: true,
    callbacks: {
      "onBatchEnd":onBatchEnd,
      "onEpochEnd":onEpochEnd
    }
  }).then(info => {
    statusElement.innerHTML = "Finished: Time = " + Math.round((performance.now()-startTime)/1000);
    let trace1 = {y:info.history.acc, name:"Training Data"};
    let trace2 = {y:info.history.val_acc, name:"Test Data"};
    layout = {title:"Accuracy",
                  autosize: false,
                  width: 600,
                  height: 600};
    Plotly.newPlot("accPlot", [trace1, trace2], layout);

    trace1 = {y:info.history.loss, name:"Training Data"};
    trace2 = {y:info.history.val_loss, name:"Test Data"};
    layout = {title:"Loss",
                  autosize: false,
                  width: 600,
                  height: 600};
    Plotly.newPlot("accPlot", [trace1, trace2], layout);

  });
  

}
