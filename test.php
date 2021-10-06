<?php
include_once('Perceptron.php');

$dataset = [];

// AND gate situations dataset
for ($i = 0; $i < 100; $i++) {
    $situations = [
        ['output' => [0], 'input' => [0, 0]],
        ['output' => [0], 'input' => [1, 0]],
        ['output' => [0], 'input' => [0, 1]],
        ['output' => [1], 'input' => [1, 1]],
    ];
    shuffle($situations);
    $dataset[] = $situations[0];
}

$perceptron = new Perceptron(Cell::SIGMOID, 1, 0.01);

$perceptron->createLayers([2, 4, 1]);

$perceptron->setInputVector($dataset[0]['input']);
$perceptron->setOutputVector($dataset[0]['output']);
$perceptron->forwardPass();
$perceptron->backPropagation();

$trainData = [];

while (abs($perceptron->getTotalError()) > $perceptron->getErrorTrashold() && $perceptron->getEpoch() < 3000) {
    foreach ($dataset as $data) {
        $perceptron->setInputVector($data['input']);
        $perceptron->setOutputVector($data['output']);
        $perceptron->forwardPass();
        $perceptron->backPropagation();
    }
}

echo 'Trained ' . $perceptron->getEpoch() . ' epochs with total error: ' . $perceptron->getTotalError() . ', with last data: ' . json_encode($data) . PHP_EOL;

$perceptron->setInputVector([0, 0]);
$perceptron->forwardPass();
echo '[0,0] : ' . json_encode($perceptron->getOutputValues()) . PHP_EOL;

$perceptron->setInputVector([0, 1]);
$perceptron->forwardPass();
echo '[0,1] : ' . json_encode($perceptron->getOutputValues()) . PHP_EOL;

$perceptron->setInputVector([1, 0]);
$perceptron->forwardPass();
echo '[1,0] : ' . json_encode($perceptron->getOutputValues()) . PHP_EOL;

$perceptron->setInputVector([1, 1]);
$perceptron->forwardPass();
echo '[1,1] : ' . json_encode($perceptron->getOutputValues()) . PHP_EOL;