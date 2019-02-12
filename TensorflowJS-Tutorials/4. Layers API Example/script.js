let x_train;
let y_train;
let x_test;
let y_test;
let model;
let history;

function get_data() {
    $.ajax({
        dataType: "json",
        url: './iris.json',
        mimeType: 'application/json',
        success: function(result) {
            const shuffelArray = arr => arr.map(a => [Math.random(), a]).sort((a, b) => a[0]-b[0]).map(a => a[1]);
            result = shuffelArray(result);
            x_train = tf.tensor2d(result.slice(0, parseInt(result.length*0.9)).map(instance => [
                instance.sepalLength, instance.sepalWidth, instance.petalLength, instance.petalWidth,
            ]))
            y_train = tf.tensor2d(result.slice(0, parseInt(result.length*0.9)).map(instance => [
                instance.species == 'setosa' ? 1:0, 
                instance.species == 'virginica' ? 1:0, 
                instance.species == 'versicolor' ? 1:0, 
            ]))
            x_test = tf.tensor2d(result.slice(parseInt(result.length*0.9)).map(instance => [
                instance.sepalLength, instance.sepalWidth, instance.petalLength, instance.petalWidth,
            ]))
            y_test = tf.tensor2d(result.slice(parseInt(result.length*0.9)).map(instance => [
                instance.species == 'setosa' ? 1:0, 
                instance.species == 'virginica' ? 1:0, 
                instance.species == 'versicolor' ? 1:0, 
            ]))
        }
    })
}

function run() {
    $.when(get_data()).then(() => {
        x_train.print()
        y_train.print()
        model = tf.sequential();
        model.add(tf.layers.dense({units:32, inputShape:[4], activation:'relu'}))
        model.add(tf.layers.dense({units:16, activation:'relu'}))
        model.add(tf.layers.dense({units:8, activation:'relu'}))
        model.add(tf.layers.dense({units:3, activation:'softmax'}))
        model.compile({loss:'categoricalCrossentropy', optimizer:'adam', metrics: ['accuracy']})
        model.fit(x_train, y_train, {epochs:100}).then((history) => {
            console.log(history.history.acc)
            model.predict(x_test).print()
        })
    })
}

run()