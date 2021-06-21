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

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    QVector<QVector<float>> trainingInputs;
    trainingInputs << QVector<float>({0, 1}) <<
                      QVector<float>({0, 0}) <<
                      QVector<float>({1, 0}) <<
                      QVector<float>({1, 1});

    QVector<float> trainingOutputs;
    trainingOutputs << 1 <<
                       0 <<
                       1 <<
                       2;

    int poolSize = 100000;
    int dataSetSize = trainingInputs.size();
    int nInputs = trainingInputs.first().size();
    QVector<Perceptron*> perceptronPool(poolSize);
    QVector<Perceptron*> offspringPool(poolSize);
    QVector<Perceptron*> superPool(poolSize + (poolSize / 2));
    for (int i = 0; i < poolSize; ++i)
    {
        perceptronPool[i] = new Perceptron();
        perceptronPool[i]->initialiseWeights(nInputs, 10);
    }


    int tournementSize = 5;
    QVector<Perceptron*> breedingPool(poolSize / tournementSize);
    QMap<int, Perceptron*> tournamentWinners;

    qsrand(QDateTime::currentMSecsSinceEpoch());

    auto shufflePool = [](QVector<Perceptron*> &pool)->void
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(pool.begin(), pool.end(), g);
    };

    auto sortLeastError = [](QVector<Perceptron*> &pool)->void
    {
        std::sort(pool.begin(), pool.end(), [&](Perceptron *a, Perceptron *b)->bool { return a->error() < b->error();});
    };

    int runs = 100000;

    float minError = 999999999;
    Perceptron bestOverallPerceptron;

    for (int run = 0; run < runs; ++ run)
    {

        // Run all perceptrons for each dataset
        for (auto *perceptron : perceptronPool)
        {
            // Need to reset the error from the last batch of runs
            perceptron->resetError();

            for (int j = 0; j < dataSetSize; ++j)
            {
                auto &inputs = trainingInputs[j];
                auto &target = trainingOutputs[j];
                perceptron->runAndSaveError(inputs, target, dataSetSize);
            }
        }

        // Sort best to worst
        sortLeastError(perceptronPool);

        // Tournament Selection
        // Take best

        Perceptron *bestPerceptron = perceptronPool.first();
        float minBreedingPoolError = bestPerceptron->error();

        if (minBreedingPoolError < minError)
        {
            minError = minBreedingPoolError;
            bestOverallPerceptron.clone(bestPerceptron);
        }

        for (int seed = 0; seed < breedingPool.size(); ++seed)
        {
            breedingPool[seed] = perceptronPool[seed];
        }

        // Push the best perceptron back intro the pool if we get worse
        if (minBreedingPoolError > minError)
        {
            breedingPool[breedingPool.size() - 1]->clone(&bestOverallPerceptron);
        }



        // Output min error
        qDebug() << "Run:" << run << " Min Error=" << minError << " CurrentBreedingPoolError: " << minBreedingPoolError;


        // Breed
        int breedingPoolSize = breedingPool.size();
        shufflePool(breedingPool);
        for (int brood = 0; brood < poolSize; ++brood)
        {
            auto *selectionA = breedingPool[brood % breedingPoolSize];
            auto *selectionB = breedingPool[(brood + 1) % breedingPoolSize];

            offspringPool[brood] = selectionA->breed(selectionB, 0.3, 1);
        }

//        // Merge the pools and choose top 50%
//        for (int i = 0; i < poolSize; ++i)
//            superPool[i] = offspringPool[i];
//        for (int i = poolSize; i < (poolSize + (poolSize / 2)); ++i)
//            superPool[i] = perceptronPool[i - poolSize];

//        // Sort the super pool
//        sortLeastError(superPool);

        // Take the top
        for (int i = 0; i < poolSize; ++i)
            perceptronPool[i]->clone(offspringPool[i]);//perceptronPool[i]->clone(superPool[i]);


        // Delete the offpsring pool (we cloned them so its ok)
        qDeleteAll(offspringPool);

        if (minError <= 0.0)
            break;
    }

    qDebug() << "Best Error" << minError;
    qDebug() << "Best Perceptron:" << "\nBias = " << bestOverallPerceptron.bias()
             << "\nWeights = " << bestOverallPerceptron.weights();

    for (int i = 0; i < trainingInputs.size(); ++i)
    {
        auto &trainingSet = trainingInputs[i];
        auto &target = trainingOutputs[i];
        qDebug() << "Input =" << trainingSet;
        qDebug() << "Target =" << target;
        qDebug() << "Output =" << bestOverallPerceptron.run(trainingSet);
        qDebug() << "\n\n\n";
    }

    //    qDeleteAll(offspring);
    //    offspring.clear();

    qDeleteAll(perceptronPool);
    perceptronPool.clear();

    return a.exec();
}
