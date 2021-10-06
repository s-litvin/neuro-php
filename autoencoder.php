<?php
include_once('Perceptron.php');

// https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/OPQMVF
// https://www.dbs.ifi.lmu.de/research/outlier-evaluation/DAMI/literature/Shuttle/

const DEBUG            = true;
const FILENAME         = 'main.txt';
const TRAIN_DATA_FILENAME = 'Shuttle_withoutdupl_norm_v01.csv';
const LOG_FILENAME     = FILENAME . '_training.log';
const OUTLIER_DATA_MARK  = 'o';
const TRAIN_DATASET_LIMIT = 700;
const TEST_DATASET_LIMIT = 2000;
const TRAIN         = true;
const CREATE_NEW    = true;
const EPOCHS        = 7000;
const TEST_ACCURACY = 0.001;

$perceptron =  CREATE_NEW ? new Perceptron(Cell::SIGMOID, 0.93, 0.01) : unserialize(file_get_contents(FILENAME));
echo CREATE_NEW ? 'New perceptron created.' . PHP_EOL : 'Saved perceptron loaded.' . PHP_EOL;

$dataset = getDataFromFile(TRAIN_DATA_FILENAME);
$trainDataset = $dataset['train'];
$testDataset = $dataset['test'];

$startTime = time();
$endTime = null;
if (TRAIN) {
    echo 'Training started...' . PHP_EOL;
    echo 'Inputs: ' . count($trainDataset[0]) . PHP_EOL;
    if (CREATE_NEW) {
        $perceptron->createLayers(
            [
                count($trainDataset[0]),
                3,
                count($trainDataset[0])
            ]
        );
    }

    $startTime = new DateTime('NOW');

    file_put_contents(LOG_FILENAME, '');

    $epoch = 0;
    while ($epoch < EPOCHS) {
        $epoch++;
        $errors = [];
        shuffle($trainDataset);
        $epochStart = time();
        foreach ($trainDataset as $data) {
            $perceptron->setInputVector($data);
            $perceptron->setOutputVector($data);
            $perceptron->forwardPass();
            $perceptron->backPropagation();
            $errors[] = abs($perceptron->getTotalError());
        }
        $str = array_sum($errors) / count($errors) . PHP_EOL;
        if (!DEBUG) {
            file_put_contents(LOG_FILENAME, $str, FILE_APPEND);
        }
        if ($epoch % 10 == 0) { // save every 10 epoch's
            echo 'Saving...' . PHP_EOL;
            file_put_contents(FILENAME, serialize($perceptron));
        }
        $epochFinished = new DateTime('NOW');
        if (DEBUG) {
            $epochTime = time() - $epochStart;
            $rest = ((EPOCHS - $epoch) * $epochTime);
            $str = '(' . $epochTime . ' sec / ' . secondsToTime($rest) . ' left). Epoch: ' . $epoch . '. Samples: ' . $perceptron->getEpoch() . '. Error: ' . $str;
            echo $str;
        }
    }
    echo 'Saving...' . PHP_EOL;
    file_put_contents(FILENAME, serialize($perceptron)); // save after finishing
    $date = new DateTime('NOW');
    echo PHP_EOL . 'STARTED ' . $startTime->format('Y-m-d H:i:s') . PHP_EOL;
    echo PHP_EOL . 'FINISHED ' . $date->format('Y-m-d H:i:s') . PHP_EOL;

}

// ------------ TEST ----------------

$success = 0;
$failure = 0;

if (DEBUG) {
    echo PHP_EOL . 'Good dataset checking' . PHP_EOL;
}

foreach ($testDataset as $data) {
    $perceptron->setInputVector(array_values($data));
    $perceptron->forwardPass();

    $out = array_values($perceptron->getOutputValues());

    $fail = false;
    foreach (array_values($data) as $index => $inputValue) {
        if (abs($out[$index] - $inputValue) > TEST_ACCURACY) {
            $fail = true;
            break;
        }
    }

    if (!$fail) {
        $success++;
    } else {
        $failure++;
        if (DEBUG) {
            echo 'F';
        }

    }
}

echo PHP_EOL . 'Failure: ' . $failure . PHP_EOL . 'Success: ' . $success . PHP_EOL;
echo 'Trained samples: ' . $perceptron->getEpoch() . PHP_EOL;

function getDataFromFile($filename)
{
    $dataset = [
        'train' => [],
        'test' => []
    ];

    if (($handle = fopen($filename, 'r')) !== false) {
        $outliersCount = 0;
        $totalRows = 0;
        $minValue = 99999;
        $maxValue = -99999;

        while (($data = fgetcsv($handle, 1000, ",")) !== false) {

            if (count($dataset['train']) >= TRAIN_DATASET_LIMIT && count($dataset['test']) >= TEST_DATASET_LIMIT) {
                break;
            }

            $dataMark = $data[count($data) - 1];
            array_pop($data);

            foreach ($data as &$value) {
                $value = (int) $value;
                if ($dataMark == OUTLIER_DATA_MARK) {
                    continue;
                }
                if ($value > $maxValue) {
                    $maxValue = $value;
                }
                if ($value < $minValue) {
                    $minValue = $value;
                }
            }

            $totalRows++;

            if ($dataMark == OUTLIER_DATA_MARK) {
                if (count($dataset['test']) < TEST_DATASET_LIMIT) {
                    $dataset['test'][] = $data;
                    $outliersCount++;
                }
                continue;
            }

            if (count($dataset['test']) < TEST_DATASET_LIMIT) {
                if (rand(0, 100) < 20) {
                    $dataset['test'][] = $data;
                    continue;
                }
            }

            if (count($dataset['train']) < TRAIN_DATASET_LIMIT) {
                $dataset['train'][] = $data;
            }
        }
        fclose($handle);

//        foreach ($dataset['train'] as &$data) {
//            foreach ($data as &$value) {
//                $value = Perceptron::normalizeInput($value, $minValue, $maxValue, 0);
//
//            }
//        }
//
//        foreach ($dataset['test'] as &$data) {
//            foreach ($data as &$value) {
//                $value = Perceptron::normalizeInput($value, $minValue, $maxValue, 0);
//            }
//        }

        echo PHP_EOL . 'Total rows: ' . $totalRows . PHP_EOL;
        echo 'Train dataset size: ' . count($dataset['train']) . PHP_EOL;
        echo 'Test dataset size: ' . count($dataset['test']) . PHP_EOL;
        echo 'Outliers: ' . $outliersCount . PHP_EOL;
        echo 'Max value: ' . $maxValue . PHP_EOL;
        echo 'Min value: ' . $minValue . PHP_EOL . PHP_EOL;
    }

    return $dataset;

}

function secondsToTime($seconds): string
{
    $dtF = new DateTime('@0');
    $dtT = new DateTime("@$seconds");
    return $dtF->diff($dtT)->format('%ad, %hh, %imin, %ssec');
}
