#include <QCoreApplication>
#include <QDateTime>
#include <QDateTime>
#include <QDebug>
#include <QThread>
#include <QVector>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <qmath.h>
#include <vector>
#include "neuralnetwork.h"
#include "opencv2/opencv.hpp"

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);
    qsrand(QDateTime::currentMSecsSinceEpoch());

    QVector<QVector<float>> trainingInputs;
    trainingInputs << QVector<float>({0, 0, 0}) <<
                      QVector<float>({0, 0, 1}) <<
                      QVector<float>({0, 1, 0}) <<
                      QVector<float>({0, 1, 1}) <<
                      QVector<float>({1, 0, 0}) <<
                      QVector<float>({1, 0, 1}) <<
                      QVector<float>({1, 1, 0}) <<
                      QVector<float>({1, 1, 1});


    QVector<float> trainingOutputs;
    trainingOutputs << 0 <<
                       1 <<
                       1 <<
                       0 <<
                       1 <<
                       0 <<
                       0 <<
                       0;

    int poolSize = 1000;
    int runs = 1000;
    int tournementSize = 10;
    float mutationRate = 0.5;
    float mutationMaxChange = 1.0;

    QVector<int> layers;
    layers << 3 << 1;

    int dataSetSize = trainingInputs.size();
    int nInputs = trainingInputs.first().size();

    QVector<NeuralNetwork*> networkPool(poolSize);
    QVector<NeuralNetwork*> offspringPool(poolSize);
    QVector<NeuralNetwork*> superPool(poolSize + (poolSize / 2));


    for (int i = 0; i < poolSize; ++i)
    {
        networkPool[i] = new NeuralNetwork();
        auto *network = networkPool[i];
        network->initialiseNetwork(nInputs, layers);
        bool debug = true;
    }



    QVector<NeuralNetwork*> breedingPool(poolSize / tournementSize);
    QMap<int, NeuralNetwork*> tournamentWinners;



    auto shufflePool = [](QVector<NeuralNetwork*> &pool)->void
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(pool.begin(), pool.end(), g);
    };

    auto sortLeastError = [](QVector<NeuralNetwork*> &pool)->void
    {
        std::sort(pool.begin(), pool.end(), [&](NeuralNetwork *a, NeuralNetwork *b)->bool { return a->error() < b->error();});
    };

    float minError = 999999999;
    NeuralNetwork bestOverallNeuralNetwork;

    cv::Mat threadPool = cv::Mat::zeros(poolSize, 1, CV_8UC1);

    for (int run = 0; run < runs; ++ run)
    {

        // Reset error from previous run
        //for (auto *network : networkPool)


        // Run all perceptrons for each dataset

        threadPool.forEach<uchar>([&](uchar &thread, const int *position)->void
        {
            int poolIndex = position[0];
           // int dataSetIndex = position[1];

            auto *network = networkPool[poolIndex];
            network->resetError();

            for (int i = 0; i < dataSetSize; ++i)
            {
            auto &inputs = trainingInputs[i];
            auto &target = trainingOutputs[i];

            network->runAndSaveError(inputs, target, dataSetSize);
            }
        });

        // Sort best to worst
        sortLeastError(networkPool);

        // Tournament Selection
        // Take best

        NeuralNetwork *bestNetwork = networkPool.first();
        float minBreedingPoolError = bestNetwork->error();

        if (minBreedingPoolError < minError)
        {
            minError = minBreedingPoolError;
            bestOverallNeuralNetwork.clone(bestNetwork);
        }

        for (int seed = 0; seed < breedingPool.size(); ++seed)
        {
            breedingPool[seed] = networkPool[seed];
        }

        // Push the best perceptron back intro the pool if we get worse
        if (minBreedingPoolError > minError)
        {
            breedingPool[breedingPool.size() - 1]->clone(&bestOverallNeuralNetwork);
        }



        // Output min error
        qDebug() << "Run:" << run << " Min Error=" << minError << " CurrentBreedingPoolError: " << minBreedingPoolError;

        qDebug() << "Best NeuralNetwork:";

        for (int i = 0; i < bestOverallNeuralNetwork.perceptrons().size(); ++i)
        {
            auto *perceptron = bestOverallNeuralNetwork.perceptrons()[i];
            qDebug() << "Perceptron" << i;
            qDebug() << "Bias =" << perceptron->bias();
            qDebug() << "Weights =" << perceptron->weights();
        }
        qDebug() << "\n";

        // Breed
        int breedingPoolSize = breedingPool.size();
        shufflePool(breedingPool);
        for (int brood = 0; brood < offspringPool.size(); ++brood)
        {
            auto *selectionA = breedingPool[brood % breedingPoolSize];
            auto *selectionB = breedingPool[(brood + 1) % breedingPoolSize];

            offspringPool[brood] = selectionA->breed(selectionB, mutationRate, mutationMaxChange);
        }

        //        // Merge the pools and choose top 50%
        //        for (int i = 0; i < poolSize; ++i)
        //            superPool[i] = offspringPool[i];
        //        for (int i = poolSize; i < (poolSize + (poolSize / 2)); ++i)
        //            superPool[i] = perceptronPool[i - poolSize];

        //        // Sort the super pool
        //        sortLeastError(superPool);

        // Take the top
        for (int i = 0; i < offspringPool.size(); ++i)
            networkPool[i]->clone(offspringPool[i]);//perceptronPool[i]->clone(superPool[i]);


        // Delete the offpsring pool (we cloned them so its ok)
        qDeleteAll(offspringPool);

        if (minError <= 0.0)
            break;
    }

    qDebug() << "Best Error" << minError;


    for (int i = 0; i < trainingInputs.size(); ++i)
    {
        auto &trainingSet = trainingInputs[i];
        auto &target = trainingOutputs[i];
        qDebug() << "Input =" << trainingSet;
        qDebug() << "Target =" << target;
        qDebug() << "Output =" << bestOverallNeuralNetwork.run(trainingSet);
        qDebug() << "\n\n\n";
    }

    for (int i = 0; i < bestOverallNeuralNetwork.perceptrons().size(); ++i)
    {
        auto *perceptron = bestOverallNeuralNetwork.perceptrons()[i];
        qDebug() << "Perceptron" << i;
        qDebug() << "Bias =" << perceptron->bias();
        qDebug() << "Weights =" << perceptron->weights();
    }

    //    qDeleteAll(offspring);
    //    offspring.clear();

    qDeleteAll(networkPool);
    networkPool.clear();

    return a.exec();
}
