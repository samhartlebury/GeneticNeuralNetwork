#include "neuralnetwork.h"
#include <QDateTime>
#include <QDebug>
#include <QtMath>

NeuralNetwork::NeuralNetwork(QObject *parent) : QObject(parent)
{

}

Perceptron::Perceptron(QObject *parent) : QObject(parent)
{
    reset();
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

    qsrand(QDateTime::currentMSecsSinceEpoch());

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

    return total;
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
    qsrand(QDateTime::currentMSecsSinceEpoch());
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
    m_weights = other->m_weights;
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
    qsrand(QDateTime::currentMSecsSinceEpoch());

    float mutationFactor = (float(qrand() % int(1000 * (max * 2))) / 1000.0) - max;
    int indexToMutate = qrand() % m_weights.size() + 1;

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



