<?php


class Cell
{
    private $targetOutput;
    private $output;
    private $input;

    public function __construct($layer)
    {
        $this->setInput(0);
        $this->setOutput(0);
        $this->setTargetOutput(null);
        $this->error = 0;
        $this->derivative = 0;
        $this->layer = $layer;
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
        return $this->error;
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
        if ($this->layer === 0) {
            $this->setOutput($inputSum);
        } else {
            $this->setOutput(1 / (1 + pow(2.718, -1 * $inputSum)));
        }

        $this->derivative = $this->getOutput() * (1 - $this->getOutput());

        return $this->getOutput();
    }

}