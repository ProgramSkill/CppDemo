namespace ArtificialIntelligence.MachineLearning.Supervised.Regression;

/// <summary>
/// Lasso回归（Least Absolute Shrinkage and Selection Operator）
/// L1正则化线性回归，可以产生稀疏解（特征选择）
/// 损失函数: MSE + α * ||w||₁
/// </summary>
public class LassoRegression
{
    private double[]? _weights;
    private double _intercept;
    private double _alpha;
    private double _learningRate;
    private int _maxIterations;

    /// <summary>
    /// 初始化Lasso回归
    /// </summary>
    /// <param name="alpha">正则化强度</param>
    /// <param name="learningRate">学习率</param>
    /// <param name="maxIterations">最大迭代次数</param>
    public LassoRegression(double alpha = 1.0, double learningRate = 0.01, int maxIterations = 1000)
    {
        _alpha = alpha;
        _learningRate = learningRate;
        _maxIterations = maxIterations;
    }

    /// <summary>
    /// 训练模型（使用坐标下降法）
    /// </summary>
    public void Fit(double[,] X, double[] y)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);

        _weights = new double[m];
        _intercept = y.Average();

        // 坐标下降优化
        for (int iter = 0; iter < _maxIterations; iter++)
        {
            // 更新每个权重
            for (int j = 0; j < m; j++)
            {
                // 计算残差（不包括当前特征的贡献）
                double[] residuals = new double[n];
                for (int i = 0; i < n; i++)
                {
                    residuals[i] = y[i] - _intercept;
                    for (int k = 0; k < m; k++)
                    {
                        if (k != j)
                            residuals[i] -= _weights[k] * X[i, k];
                    }
                }

                // 计算相关性
                double correlation = 0;
                double normSquared = 0;
                for (int i = 0; i < n; i++)
                {
                    correlation += X[i, j] * residuals[i];
                    normSquared += X[i, j] * X[i, j];
                }

                // 软阈值更新（Soft Thresholding）
                if (normSquared > 0)
                {
                    if (correlation > _alpha)
                        _weights[j] = (correlation - _alpha) / normSquared;
                    else if (correlation < -_alpha)
                        _weights[j] = (correlation + _alpha) / normSquared;
                    else
                        _weights[j] = 0;
                }
            }

            // 更新截距
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                double pred = _intercept;
                for (int j = 0; j < m; j++)
                    pred += _weights[j] * X[i, j];
                sum += y[i] - pred;
            }
            _intercept += sum / n;
        }
    }

    /// <summary>
    /// 预测
    /// </summary>
    public double[] Predict(double[,] X)
    {
        if (_weights == null)
            throw new InvalidOperationException("模型未训练");

        int n = X.GetLength(0);
        int m = X.GetLength(1);
        double[] predictions = new double[n];

        for (int i = 0; i < n; i++)
        {
            predictions[i] = _intercept;
            for (int j = 0; j < m; j++)
                predictions[i] += _weights[j] * X[i, j];
        }

        return predictions;
    }

    /// <summary>
    /// 获取非零权重的数量（特征选择结果）
    /// </summary>
    public int GetNonZeroWeightsCount()
    {
        if (_weights == null)
            throw new InvalidOperationException("模型未训练");

        return _weights.Count(w => Math.Abs(w) > 1e-10);
    }
}
