// Creating dataset
const x = tf.randomUniform([100], -1, 1)
const y = x.mul(0.5).add(tf.randomUniform([100], -0.1, 0.1))

// Creating variables
m = tf.variable(tf.scalar(Math.random()*2-1))
b = tf.variable(tf.scalar(Math.random()*2-1))

// Specifying a learning rate and an optimizer
const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate)

// Training model for 50 epochs
for(let i = 0; i < 50; i++) {
    optimizer.minimize(() => loss(predict(x), y))
}

// Printing our weights (slope, intercept)
m.print()
b.print()

// Mean Squared Error Loss
function loss(pred, label) {
    return pred.sub(label).square().mean()
}

function predict(xs) {
    // y = mx+b
    const ys = xs.mul(m).add(b)
    return ys
}

function visualize() {
    const xArr = x.dataSync();
    const yArr = y.dataSync();
    const ys = predict(x);
    const lineY = ys.dataSync()

    let dataset = [];
    let line = [];

    for(let i = 0; i < xArr.length; i++){
        line.push({'x': xArr[i], 'y': lineY[i]});
        dataset.push({'x': xArr[i], 'y': yArr[i]});
    }

    let ctx = document.getElementById('myChart').getContext('2d');

    let myChart = new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Dataset',
                data: dataset,
                backgroundColor: 'rgb(255, 99, 132)',
                borderColor: 'rgb(255, 99, 132)',
            },{
                label: 'Regression Line',
                data: line,
                backgroundColor: 'rgb(66, 134, 244)',
                borderColor: 'rgb(39, 96, 188)',
            }]
        }
    })
}

visualize()