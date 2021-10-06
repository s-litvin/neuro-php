<?php


class Cell
{
    const RELU = 'relu';
    const LEAKYRELU = 'leakyrelu';
    const SIGMOID = 'sigmoid';

    private $targetOutput;
    private $output;
    private $input;
    private $isBias;
    private $activation;

    public function __construct($layer, $isBias = false, $activation = self::SIGMOID)
    {
        $this->setInput(0);
        $this->setOutput(0);
        $this->setTargetOutput(null);
        $this->error = 0;
        $this->derivative = 0;
        $this->layer = $layer;
        $this->isBias = $isBias;
        $this->activation = $activation;
    }

    /**
     * @return int
     */
    public function getInput()
    {
        return $this->input;
    }

    public function setInput($input)
    {
        $this->input = $input;
    }

    public function setOutput($output)
    {
        $this->output = $output;
    }

    public function setTargetOutput($targetOutput)
    {
        $this->targetOutput = $targetOutput;
    }

    public function setError($error)
    {
        $this->error = $error;
    }

    public function getError()
    {
        return is_nan($this->error) ? 0 : $this->error;
    }

    public function getDerivative()
    {
        return $this->derivative;
    }

    public function getTargetOutput()
    {
        return $this->targetOutput;
    }

    public function calcError()
    {
        // $this->error = 0.5 * Math.pow($this->targetOutput - $this->getOutput(), 2);
        $this->error = $this->targetOutput - $this->getOutput();
    }


    public function getOutput()
    {
        return $this->output;
    }

    public function calcOutput($inputSum)
    {
        if ($this->isBias()) {
            $this->setOutput(1);
        } else if ($this->layer === 0) {
            $this->setOutput($inputSum);
        } else {
            $this->setOutput($this->calcActivation($inputSum));
        }

        $this->derivative = $this->calcDerivative();

        return $this->getOutput();
    }

    protected function calcActivation($inputSum)
    {
        switch ($this->activation) {
            case self::RELU:
                $calcSum = $inputSum < 0 ? 0 : $inputSum;
                break;
            case self::LEAKYRELU:
                $calcSum = $inputSum < 0 ? 0.01 * $inputSum : $inputSum;
                break;
            case self::SIGMOID:
            default:
                $calcSum = 1 / (1 + pow(2.718, -1 * $inputSum));
        }

        return $calcSum;
    }

    protected function calcDerivative()
    {
        switch ($this->activation) {
            case self::RELU:
            case self::LEAKYRELU:
                return $this->getOutput() < 0 ? 0 : 1;
            case self::SIGMOID:
            default:
                return $this->getOutput() * (1 - $this->getOutput());
        }
    }

    public function isBias()
    {
        return $this->isBias;
    }

}