namespace ArtificialIntelligence.MachineLearning.Supervised.Classification;

/// <summary>
/// 逻辑回归分类器
/// 用于二分类问题，使用Sigmoid函数将线性输出映射到[0,1]区间
/// </summary>
public class LogisticRegression
{
    private double[]? _weights;
    private double _intercept;
    private double _learningRate;
    private int _maxIterations;

    public LogisticRegression(double learningRate = 0.01, int maxIterations = 1000)
    {
        _learningRate = learningRate;
        _maxIterations = maxIterations;
    }

    /// <summary>
    /// 训练模型（使用梯度下降）
    /// </summary>
    public void Fit(double[,] X, int[] y)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);

        _weights = new double[m];
        _intercept = 0.0;

        // 梯度下降优化
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            double[] predictions = PredictProba(X);

            // 计算梯度
            double[] gradW = new double[m];
            double gradB = 0.0;

            for (int i = 0; i < n; i++)
            {
                double error = predictions[i] - y[i];
                gradB += error;
                for (int j = 0; j < m; j++)
                {
                    gradW[j] += error * X[i, j];
                }
            }

            // 更新参数
            _intercept -= _learningRate * gradB / n;
            for (int j = 0; j < m; j++)
            {
                _weights[j] -= _learningRate * gradW[j] / n;
            }
        }
    }

    /// <summary>
    /// 预测概率
    /// </summary>
    public double[] PredictProba(double[,] X)
    {
        if (_weights == null)
            throw new InvalidOperationException("模型未训练");

        int n = X.GetLength(0);
        int m = X.GetLength(1);
        double[] probabilities = new double[n];

        for (int i = 0; i < n; i++)
        {
            double z = _intercept;
            for (int j = 0; j < m; j++)
            {
                z += _weights[j] * X[i, j];
            }
            probabilities[i] = Sigmoid(z);
        }

        return probabilities;
    }

    /// <summary>
    /// 预测类别（阈值0.5）
    /// </summary>
    public int[] Predict(double[,] X)
    {
        double[] probabilities = PredictProba(X);
        int[] predictions = new int[probabilities.Length];

        for (int i = 0; i < probabilities.Length; i++)
        {
            predictions[i] = probabilities[i] >= 0.5 ? 1 : 0;
        }

        return predictions;
    }

    private static double Sigmoid(double z)
    {
        return 1.0 / (1.0 + Math.Exp(-z));
    }
}
