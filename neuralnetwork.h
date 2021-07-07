#ifndef NEUTALNETWORK_H
#define NEUTALNETWORK_H

#include <QMap>
#include <QObject>
#include <mutex>
#include <shared_mutex>

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
    bool sigmoidActivationEnabled();
    void setSigmoidActivationEnabled(bool enabled);
    bool roundOutput();
    void setRoundOutput(bool enabled);

    float bias() const;
    QVector<float> weights() const;

    void mutate(float max);

    void runAndSaveError(QVector<float> inputs, float target, int divider = 1);

    void resetError();

    void setNetworkParent(Perceptron *parent);
    void setNetworkParents(QVector<Perceptron*> parents);

    QVector<Perceptron*> networkParents();

    Perceptron *networkParent();

    float networkRun(QVector<float> inputs);


private:
    QVector<float> m_weights;
    QVector<float> m_exponents;
    float m_bias;
    float m_error = 0;
    Perceptron *m_networkParent;
    QVector<Perceptron*> m_networkParents;
    bool m_sigmoidActivationEnabled = true;
    bool m_roundOutput = false;
};


class NeuralNetwork : public QObject
{
    Q_OBJECT

public:
    explicit NeuralNetwork(QObject *parent = nullptr);
    ~NeuralNetwork();
    Perceptron *createPerceptron();
    void initialiseNetwork(int inputs, QVector<int> layers, bool sigmoidOutputLayer = true);

    QVector<Perceptron*> perceptrons();


    float run(QVector<float> inputs);
    QVector<float> runMultiOutput(QVector<float> inputs);

    void runAndSaveError(QVector<float> inputs, float target, int divider = 1);
    void runMultiOutputAndSaveError(QVector<float> inputs, QVector<float> targets, int divider = 1);

    float error();
    void setError(float error);
    void resetError();

    void clone(NeuralNetwork *other);
    NeuralNetwork *breed(NeuralNetwork *mate, float mutationRate = 0.1, float amount = 1.0);

    void crossOverBreed(NeuralNetwork *child, NeuralNetwork *mateA, NeuralNetwork *mateB, float mutationRate = 0.1, float amount = 1.0);

    QByteArray drawNetwork();



private:
    QVector<Perceptron*> m_perceptrons;

    QMap<int, Perceptron*> m_networkMap;

    QVector<int> m_layers;
    int m_inputs;
    float m_error;

    mutable std::shared_mutex m_errorMutex;

};

#endif // NEUTALNETWORK_H
