const data1 = [];
const data2 = [];

// Fill to arrays with random values from 0-50
for(let i = 0; i < 50; i++) {
    data1.push(Math.random()*10);
    data2.push(Math.random()*10);
}

// Create 1d Tensors
const tensor = tf.tensor1d(data1);
console.log(tensor);
tensor.print();

// Create a scalar(number)
const scalar = tf.scalar(3);
scalar.print();

// Specify a data-type
const tensor2 = tf.tensor1d(data1, dtype='int32');
tensor2.print();

// Create 2d Tensor
const tensor3 = tf.tensor2d(data1, [10, 5]); 
tensor3.print();

// Create tensor filled with zeros
const zerosTensor = tf.zeros([3, 2]);
zerosTensor.print();

// Create tensor filled with ones
const onesTensor = tf.ones([3, 2]);
onesTensor.print();

// Create tensor filled with scalars in a specific range
const rangeTensor = tf.range(0, 10, 2);
rangeTensor.print();

// Create a identity matrix
const identityTensor = tf.eye(3, 3);
identityTensor.print();

// Operations
const a = tf.tensor1d(data1, dtype='int32');
const b = tf.tensor1d(data2, dtype='int32');

// Add
console.log('Add:')
a.print();
b.print();
a.add(b).print();

// Multiply
console.log('Multiply:')
a.print();
b.print();
a.mul(b).print();

const c = tf.tensor2d([1, 2, 3, 4, 5, 6], [3, 2])
const d = tf.tensor2d([1, 2, 3, 4, 5, 6], [2, 3])
c.print()
d.print()
c.matMul(d).print()