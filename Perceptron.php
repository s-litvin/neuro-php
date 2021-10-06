<?php
include_once('Cell.php');

class Perceptron
{
    /**
     * @var float|mixed
     */
    private $learningRate;
    /**
     * @var int
     */
    private $totalError;
    /**
     * @var float|mixed
     */
    private $errorTrashold;
    /**
     * @var array
     */
    private $neurones;
    /**
     * @var array
     */
    private $layers;
    private $epoch = 0;
    private $outputValues;

    public function __construct($activationFunction = Cell::SIGMOID, $learningRate = 0.98, $errorTrashhold = 0.0001)
    {
        $this->activation = $activationFunction;
        $this->neurones = [];
        $this->layers = [];
        $this->totalError = 0;
        $this->learningRate = $learningRate;
        $this->errorTrashold = $errorTrashhold;
    }

    public function addNeuron(Cell $cell, $id, $layer, $bias = 0)
    {
        $this->neurones[$id] = [
            'id' => $id,
            'cell' => $cell,
            'links' => [],
            'layer' => $layer,
        ];

        $this->indexLayers();
    }

    /**
     * @return float|mixed
     */
    public function getLearningRate()
    {
        return $this->learningRate;
    }

    /**
     * @return int
     */
    public function getEpoch()
    {
        return $this->epoch;
    }

    /**
     * @return float|mixed
     */
    public function getErrorTrashold()
    {
        return $this->errorTrashold;
    }

    /**
     * @return int
     */
    public function getTotalError()
    {
        return $this->totalError;
    }

    public function indexLayers()
    {
        $this->layers = [];
        foreach ($this->neurones as $neuron) {
            $this->layers[] = $neuron['layer'];
        }
        $this->layers = array_unique($this->layers);
        sort($this->layers);
    }

    public function updateNeuron($id, $neuron)
    {
        $this->neurones[$id] = $neuron;
        $this->indexLayers();
    }

    public function getNeuronsByLayer($layerNumber)
    {
        return array_filter(
            $this->neurones,
            function ($neuron) use ($layerNumber) {
                return $neuron['layer'] == $layerNumber;
            }
        );
    }

    public function getNeuronLinks($neuron, $linkType)
    {
        $neurones = [];
        $links = array_filter(
            $neuron['links'],
            function ($link) use ($linkType) {
                return $link['type'] == $linkType;
            }
        );

        foreach ($links as $link) {
            $neurones[] = [
                'neuron' => $this->neurones[$link['id']],
                'weight' => $link['weight'],
            ];
        }

        return $neurones;
    }

    public function link($id1, $id2, $weight = null)
    {
        if ($weight === null) {
            $weight = rand(1, 100) / 100;
            if (rand(0, 1) == 0) {
                $weight *= -1;
            }
        }

        $n1 = $this->neurones[$id1];
        $n2 = $this->neurones[$id2];

        $n1['links'][] = ['id' => $id2, 'weight' => $weight, 'type' => 'right'];
        $n2['links'][] = ['id' => $id1, 'weight' => $weight, 'type' => 'left'];

        $n1['links'] = array_values($n1['links']);
        $n2['links'] = array_values($n2['links']);

        $this->updateNeuron($id1, $n1);
        $this->updateNeuron($id2, $n2);
    }

    public function unlink($id1, $id2)
    {
        $n1 = $this->neurones[$id1];
        $n2 = $this->neurones[$id2];
        $n1['links'] = array_filter(
            $n1['links'],
            function ($link) use ($id2) {
                return $link['id'] != $id2;
            }
        );
        $n2['links'] = array_filter(
            $n2['links'],
            function ($link) use ($id1) {
                return $link['id'] != $id1;
            }
        );
        $this->updateNeuron($id1, $n1);
        $this->updateNeuron($id2, $n2);
    }

    public function linkAll()
    {
        foreach ($this->layers as $layer) {

            $neuronesLeft = $this->getNeuronsByLayer($layer);
            $neuronesRight = $this->getNeuronsByLayer($layer + 1);

            foreach ($neuronesLeft as $leftNeuron) {
                foreach ($neuronesRight as $rightNeuron) {
                    if ($rightNeuron['cell']->isBias()) {
                        continue;
                    }
                    $this->link($leftNeuron['id'], $rightNeuron['id']);
                }
            }
        }
    }

    public function createLayers($neuronsCountArray, $linkAutomatically = true)
    {
        foreach ($neuronsCountArray as $layer => $neuronsNumber) {
            for ($number = 0; $number < $neuronsNumber; $number++) {
                $letter = 'h';
                if ($layer == 0) {
                    $letter = 'x';
                } else if ($layer == count($neuronsCountArray) - 1) {
                    $letter = 'y';
                }
                $this->addNeuron(new Cell($layer, false, $this->activation), $letter . $layer . $number, $layer);
            }

            if ($layer != count($neuronsCountArray) - 1) {
                // Bias neuron
                $this->addNeuron(new Cell($layer, true, $this->activation), 'b' . $layer . ($number + 1), $layer);
            }
        }

        if ($linkAutomatically) {
            $this->linkAll();
        }
    }

    public function setInputVector($inputsArray)
    {
        $firstLayerNeurones = $this->getNeuronsByLayer($this->layers[0]);

        $key = 0;
        foreach ($firstLayerNeurones as $firstLayerNeuron) {
            if ($firstLayerNeuron['cell']->isBias()) {
                continue;
            }
            $firstLayerNeuron['cell']->setInput($inputsArray[$key++]);
            $this->updateNeuron($firstLayerNeuron['id'], $firstLayerNeuron);
        }
    }

    public function setOutputVector($outputsArray)
    {
        $lastLayerNeurones = $this->getNeuronsByLayer($this->layers[count($this->layers) - 1]);

        $key = 0;
        foreach ($lastLayerNeurones as $lastLayerNeuron) {
            $lastLayerNeuron['cell']->setTargetOutput($outputsArray[$key++]);
            $this->updateNeuron($lastLayerNeuron['id'], $lastLayerNeuron);
        }
    }

    public function forwardPass()
    {
        $lastLayer = end($this->layers);
        $this->outputValues = [];

        foreach ($this->layers as $layer) {

            $neurones = $this->getNeuronsByLayer($layer);

            foreach ($neurones as $neuron) {

                $inputSum = $neuron['cell']->getInput();
                $leftLinks = $this->getNeuronLinks($neuron, 'left');

                foreach ($leftLinks as $leftLink) {
                    $inputSum += $leftLink['neuron']['cell']->getOutput() * $leftLink['weight'];
                }

                $neuron['cell']->calcOutput($inputSum);

                if ($layer == $lastLayer) {
                    $this->outputValues[$neuron['id']] = $neuron['cell']->getOutput();
                }
            }
        }
    }

    public function calcErrors()
    {

        $this->totalError = 0;
        $this->epoch++;

        for ($li = count($this->layers) - 1; $li >= 0; $li--) {
            $neurones = $this->getNeuronsByLayer($this->layers[$li]);

            foreach ($neurones as $index => $neuron) {
                $rightLinks = $this->getNeuronLinks($neuron, 'right');

                if (!$rightLinks) { // если это последний слой, то ошибка вычисляется разницей ожидания и выхода
                    $neuron['cell']->setError($neuron['cell']->getTargetOutput() - $neuron['cell']->getOutput());
                } else { // если слой скрытый, то ошибка - сумма ошибок правых узлов умноженных на веса.
                    $errorsSum = 0;

                    foreach ($rightLinks as $rightLink) {
                        $errorsSum += $rightLink['neuron']['cell']->getError() * $rightLink['weight'];
                    }

                    $neuron['cell']->setError($errorsSum);
                }

                if ($li === count($this->layers) - 1) {
                    $this->totalError += $neuron['cell']->getError();
                }
            }
        }
    }

    public function updateWeights()
    {
        foreach ($this->layers as $layer) {

            $neurones = $this->getNeuronsByLayer($layer);

            foreach ($neurones as $neuron) {

                $rightLinks = $this->getNeuronLinks($neuron, 'right');

                foreach ($rightLinks as $rightLink) {
                    $rightNeuron = $rightLink['neuron'];
                    $newWeight =
                        $rightLink['weight'] +
                        $rightNeuron['cell']->getDerivative() *
                        $rightNeuron['cell']->getError() *
                        $neuron['cell']->getOutput() *
                        $this->getLearningRate();
                    $this->unlink($neuron['id'], $rightNeuron['id']);
                    $this->link($neuron['id'], $rightNeuron['id'], $newWeight);
                }
            }
        }
    }

    public function backPropagation()
    {
        $this->calcErrors();
        $this->updateWeights();

        return true;
    }

    /**
     * @return mixed
     */
    public function getOutputValues()
    {
        return $this->outputValues;
    }

    public static function normalizeInput($value, $min, $max, $toMin = -1, $toMax = 1)
    {
        return (($value - $min) * ($toMax - $toMin)) / ($max - $min) + $toMin;
    }
}