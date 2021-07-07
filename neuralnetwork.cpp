#include "neuralnetwork.h"
#include <QDateTime>
#include <QDebug>
#include <QtMath>

NeuralNetwork::NeuralNetwork(QObject *parent) : QObject(parent)
{
    resetError();
}

NeuralNetwork::~NeuralNetwork()
{
    qDeleteAll(m_perceptrons);
    m_perceptrons.clear();
}

Perceptron *NeuralNetwork::createPerceptron()
{
    Perceptron *perceptron = new Perceptron();
    m_perceptrons << perceptron;
    return perceptron;
}

void NeuralNetwork::initialiseNetwork(int inputs, QVector<int> layers, bool sigmoidOutputLayer)
{
    m_inputs = inputs;
    m_layers = layers;

    for (int i = 0; i < layers.size(); ++i)
    {
        for (int j = 0; j < layers[i]; ++j)
        {
            auto *perceptron = createPerceptron();

            m_networkMap.insertMulti(i, perceptron);

            if (i == 0)
            {
                perceptron->initialiseWeights(m_inputs, 1000.0);
                continue;
            }

            auto parents = m_networkMap.values(i - 1);
            perceptron->initialiseWeights(parents.size(), 1000.0);
            perceptron->setNetworkParents(parents.toVector());

            // Disable sigmoid activation for Output perceptron layer so it can be used for regression problems
            if (i == layers.size() - 1 && !sigmoidOutputLayer)
            {
                perceptron->setSigmoidActivationEnabled(false);
                // perceptron->setRoundOutput(true);
            }
        }
    }
}

QVector<Perceptron *> NeuralNetwork::perceptrons()
{
    return m_perceptrons;
}

float NeuralNetwork::run(QVector<float> inputs)
{
    return m_perceptrons.last()->networkRun(inputs);
}

QVector<float> NeuralNetwork::runMultiOutput(QVector<float> inputs)
{
    int outputLayerIndex = m_layers.size() - 1;
    auto outputLayer = m_networkMap.values(outputLayerIndex);

    QVector<float> outputs;
    for (auto *perceptron : outputLayer)
    {
        float result = perceptron->networkRun(inputs);
        outputs << result;
    }

    return outputs;
}

void NeuralNetwork::runAndSaveError(QVector<float> inputs, float target, int divider)
{
    m_error += qPow(run(inputs) - target, 2) / float(divider);

    // float newError = qPow(m_perceptrons.last()->networkRun(inputs) - target, 2);

    // setError(newError);

}

void NeuralNetwork::runMultiOutputAndSaveError(QVector<float> inputs, QVector<float> targets, int divider)
{
    auto outputs = runMultiOutput(inputs);

    for (int i = 0; i < outputs.size(); ++i)
    {
        auto output = outputs[i];
        auto target = targets[i];

        m_error += qPow(output - target, 2) / float(divider);
    }
}

float NeuralNetwork::error()
{
    std::shared_lock<std::shared_mutex> lock(m_errorMutex);
    return m_error;
}

void NeuralNetwork::setError(float error)
{
    std::unique_lock<std::shared_mutex> lock(m_errorMutex);
    m_error += error;
}

void NeuralNetwork::resetError()
{
    std::unique_lock<std::shared_mutex> lock(m_errorMutex);
    m_error = 0.0;
}

void NeuralNetwork::clone(NeuralNetwork *other)
{
    m_error = other->error();

    qDeleteAll(m_perceptrons);
    m_perceptrons.clear();
    m_networkMap.clear();

    initialiseNetwork(other->m_inputs, other->m_layers);

    if (m_perceptrons.size() != other->m_perceptrons.size())
        Q_ASSERT(false);

    for (int i = 0; i < m_perceptrons.size(); ++i)
    {
        m_perceptrons[i]->clone(other->m_perceptrons[i]);
    }
}


NeuralNetwork *NeuralNetwork::breed(NeuralNetwork *mate, float mutationRate, float amount)
{
    NeuralNetwork *child = new NeuralNetwork();

    child->initialiseNetwork(m_inputs, m_layers);

    // for (int i = 0; i < m_perceptrons.size(); ++i)
    //  {
    // auto *offspring = m_perceptrons[i]->breed(mate->m_perceptrons[i], mutationRate, amount);
    //  child->m_perceptrons[i]->clone(offspring);
    // delete offspring;

    // }

    crossOverBreed(child, this, mate, mutationRate, amount);

    return child;
}

void NeuralNetwork::crossOverBreed(NeuralNetwork *child, NeuralNetwork *mateA, NeuralNetwork *mateB, float mutationRate, float amount)
{
    int crossoverPoint = qrand() % mateA->m_perceptrons.size();

    for (int i = 0; i < mateA->m_perceptrons.size(); ++i)
        child->m_perceptrons[i]->clone(i < crossoverPoint ? mateA->m_perceptrons[i] : mateB->m_perceptrons[i]);


    for (int i = 0; i < mateA->m_perceptrons.size(); ++i)
        if (!(qrand() % int((1.0 / mutationRate) + 0.5)))
        {
            auto *randomPerceptron = child->m_perceptrons[qrand() % child->m_perceptrons.size()];
            randomPerceptron->mutate(amount);
        }
}

QByteArray NeuralNetwork::drawNetwork()
{
    QByteArray output;
    for (auto level : m_networkMap.keys())
    {
        output += "Level " + QString::number(level + 1) + "\n";
        int perceptronCount = 1;
        for (auto *perceptron : m_networkMap.values(level))
        {
            output += "Perceptron" + QString::number(perceptronCount) + "\n";
            output += "Bias =" + QString::number(perceptron->bias()) + "\n";
            output += "Weights = ";
            for (auto weight : perceptron->weights())
                output += QString::number(weight) + ", ";

            output += "\n";
            perceptronCount++;
        }

        output += "\n\n";
    }

    return output;
}


Perceptron::Perceptron(QObject *parent) : QObject(parent)
{
    reset();
    m_sigmoidActivationEnabled = true;
}

void Perceptron::initialiseWeights(int nInputs, int limit)
{
    m_weights = QVector<float>(nInputs);
    m_bias = 1.0;
    for (auto &weight : m_weights)
        weight = 1.0;

    for (int i = 0; i < m_weights.size() + 1; ++i)
        mutate(limit);
}

void Perceptron::reset()
{
    m_weights.clear();
    m_bias = 0.0;
}

void Perceptron::fit(QVector<QVector<float> > inputs, QVector<float> outputs)
{
    initialiseWeights(inputs.first().size());

    int runs = 10000;
    float trainingRate = 0.1;
    float smallestError = 100;

    for (int i = 0; i < runs; ++i)
    {
        int errors = 0;

        int j = qrand() % inputs.size();

        float update = trainingRate * (outputs[j] - run(inputs[j]));
        m_bias = update;
        for (int w = 0; w < m_weights.size(); ++w)
        {
            m_weights[w] += update * inputs[j][w];
        }

        // errors += update != 0 ? 1 : 0;
        errors += qAbs(update);

        if (errors < smallestError)
        {
            qDebug() << "Run number " << QString::number(i);
            smallestError = errors;
            qDebug() << errors << "\n\n";
        }
    }
}

float Perceptron::liveFit(float actual, float output, QVector<float> inputs)
{
    float trainingRate = 0.1;
    float update = trainingRate * (actual - output);
    m_bias = update;

    for (int w = 0; w < m_weights.size(); ++w)
    {
        m_weights[w] += update * inputs[w];
    }

    return update;
}



float Perceptron::run(QVector<float> inputs)
{
    if (inputs.size() != m_weights.size())
    {
        Q_ASSERT(false);
        return 0;
    }

    float total = m_bias;

    for (int i = 0; i < inputs.size(); ++i)
    {
        total += inputs[i] * m_weights[i];
    }

    float resultSigmoided = sigmoid(total);

    float finalResult = total;

    if (m_sigmoidActivationEnabled)
        finalResult = resultSigmoided;
    if (m_roundOutput)
        finalResult = int(finalResult + 0.5);

    return finalResult;
}

void Perceptron::setWeights(const QVector<float> &weights)
{
    m_weights = weights;
}

float Perceptron::error()
{
    return m_error;
}

Perceptron* Perceptron::breed(Perceptron *mate, float mutationRate, float amount)
{
    auto mineOrTheirs = []()->bool { return qrand() % 2;};

    auto *child = new Perceptron();
    child->initialiseWeights(m_weights.size());

    int crossoverPoint = qrand() % m_weights.size();

    for (int i = 0; i < m_weights.size(); ++i)
    {
        const auto &myWeight = m_weights[i];
        const auto &theirWeight = mate->m_weights[i];

        child->m_weights[i] = i < crossoverPoint ? myWeight : theirWeight;
    }

    mineOrTheirs() ? child->m_bias = m_bias : mate->m_bias;

    if (!(qrand() % int((1.0 / mutationRate) + 0.5)))
    {
        child->mutate(amount);
    }

    return child;
}

float Perceptron::sigmoid(float x)
{
    return 1.0 / (1.0 + qExp(-x));
}

void Perceptron::clone(Perceptron *other)
{
    m_bias = other->m_bias;
    m_weights = QVector<float>(other->m_weights);
    m_error = other->m_error;
}

bool Perceptron::operator==(const Perceptron &other)
{
    if (m_weights.size() != other.m_weights.size())
        return false;

    if (m_bias != other.m_bias)
        return false;

    for (int i = 0; i < m_weights.size(); ++i)
    {
        if (m_weights[i] != other.m_weights[i])
            return false;
    }

    return true;
}

bool Perceptron::sigmoidActivationEnabled()
{
    return m_sigmoidActivationEnabled;
}

void Perceptron::setSigmoidActivationEnabled(bool enabled)
{
    m_sigmoidActivationEnabled = enabled;
}

bool Perceptron::roundOutput()
{
    return m_roundOutput;
}

void Perceptron::setRoundOutput(bool enabled)
{
    m_roundOutput = enabled;
}

float Perceptron::bias() const
{
    return m_bias;
}

QVector<float> Perceptron::weights() const
{
    return m_weights;
}

void Perceptron::mutate(float max)
{
    float randomFloat = static_cast<float>(qrand()) / static_cast<float>(RAND_MAX);
    float mutationFactor = randomFloat * max;

    if (qrand() % 2)
        mutationFactor *= -1.0;

    int indexToMutate = qrand() % (m_weights.size() + 1);

    if (indexToMutate == m_weights.size())
        m_bias *= mutationFactor;
    else
        m_weights[indexToMutate] *= mutationFactor;
}

void Perceptron::runAndSaveError(QVector<float> inputs, float target, int divider)
{
    // Sum Square Error
    //  m_error += qPow(run(inputs) - target, 2);
    // Mean Square Error
    m_error += qPow(run(inputs) - target, 2) / float(divider);
}

void Perceptron::resetError()
{
    m_error = 0;
}

void Perceptron::setNetworkParent(Perceptron *parent)
{
    m_networkParent = parent;
}

void Perceptron::setNetworkParents(QVector<Perceptron *> parents)
{
    m_networkParents = parents;
}

QVector<Perceptron *> Perceptron::networkParents()
{
    return m_networkParents;
}

Perceptron *Perceptron::networkParent()
{
    return m_networkParent;
}

float Perceptron::networkRun(QVector<float> inputs)
{
    if (m_networkParents.isEmpty())
        return run(inputs);

    QVector<float> parentsOutputs;

    for (auto *parent : m_networkParents)
        parentsOutputs << parent->networkRun(inputs);

    return run(parentsOutputs);
}



