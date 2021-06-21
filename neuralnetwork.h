#ifndef NEUTALNETWORK_H
#define NEUTALNETWORK_H

#include <QObject>

class Perceptron : public QObject
{
    Q_OBJECT
public:
    explicit Perceptron(QObject *parent = nullptr);

    void initialiseWeights(int nInputs, int limit = 100);
    void reset();

    void fit(QVector<QVector<float>> inputs, QVector<float> outputs);

    float liveFit(float actual, float output, QVector<float> inputs);

    float run(QVector<float> inputs);

    void setWeights(const QVector<float> &weights);

    float error();

    Perceptron *breed(Perceptron *mate, float mutationRate = 0.1, float amount = 1.0);

    float sigmoid(float x);

    void initRandomWeights();

    void clone(Perceptron *other);
    bool operator==(const Perceptron &other);

    float bias() const;
    QVector<float> weights() const;

    void mutate(float max);

    void runAndSaveError(QVector<float> inputs, float target, int divider = 1);

    void resetError();


private:
    QVector<float> m_weights;
    float m_bias;
    float m_error = 0;
};


class NeuralNetwork : public QObject
{
    Q_OBJECT
public:
    explicit NeuralNetwork(QObject *parent = nullptr);


};

#endif // NEUTALNETWORK_H
