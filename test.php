<?php
include_once('Perceptron.php');

$perceptron = new Perceptron();

$perceptron->createLayers([5, 5, 5, 3]);

$perceptron->setInputVector([0.61, 0.12, 0.45, 0.23, 0.29]);
$perceptron->setOutputVector([0.91, 0.1, 0.2]);

$perceptron->forwardPass();
$perceptron->backPropagation();

while (abs($perceptron->getTotalError()) > $perceptron->getErrorTrashold() && $perceptron->getEpoch() < 100) {
    $perceptron->forwardPass();
    $perceptron->backPropagation();
}

echo '#' . $perceptron->getEpoch() . '. ' . $perceptron->getTotalError();