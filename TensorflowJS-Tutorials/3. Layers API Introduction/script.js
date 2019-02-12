// Creating a perceptron
const model1 = tf.sequential();

const hidden = tf.layers.dense({
    units: 8,
    activation: 'relu',
    inputShape: 2
});
model1.add(hidden);

const output = tf.layers.dense({
    units: 1,
    activation: 'sigmoid'
});
model1.add(output);


// Another way to write the same network
const model2 = tf.sequential();
model2.add(tf.layers.dense({units:8, activation:'relu', inputShape:2}));
model2.add(tf.layers.dense({units:1, activation:'sigmoid'}));


// Creating a multilayer perceptron
const model3 = tf.sequential();
model3.add(tf.layers.dense({units:32, activation:'relu', inputShape:2}));
model3.add(tf.layers.dense({units:16, activation:'relu'}));
model3.add(tf.layers.dense({units:8, activation:'relu'}));
model3.add(tf.layers.dense({units:1, activation:'sigmoid'}));


// Creating a convolutional neural network
const model4 = tf.sequential();
model4.add(tf.layers.conv2d({filters:16, kernelSize:5, strides:1, activation:'relu', inputShape:[28, 28, 1]}));
model4.add(tf.layers.maxPooling2d({poolSize:[2, 2], strides:[1, 1]}));
model4.add(tf.layers.conv2d({filters:8, kernelSize:5, strides:1, activation:'relu'}));
model4.add(tf.layers.maxPooling2d({poolSize:[2, 2], strides:[1, 1]}));
model4.add(tf.layers.dropout(0.2))
model4.add(tf.layers.flatten());
model4.add(tf.layers.dense({units:128, activation:'relu'}))
model4.add(tf.layers.dense({units:10, activation:'softmax'}))


// Creating an recurrent neural network
const model5 = tf.sequential()
model5.add(tf.layers.lstm({units:96, returnSequences:true, inputShape:[16, 1]}))
model5.add(tf.layers.dropout(0.2))
model5.add(tf.layers.lstm({units:96, returnSequences:true}))
model5.add(tf.layers.dropout(0.2))
model5.add(tf.layers.lstm({units:96}))
model5.add(tf.layers.dropout(0.2))
model5.add(tf.layers.dense({units:1}))