namespace ArtificialIntelligence.MachineLearning.Supervised.Regression;

/// <summary>
/// 岭回归（Ridge Regression）- L2正则化线性回归
/// 损失函数: MSE + α * ||w||²
/// 用于防止过拟合，特别适用于特征间存在多重共线性的情况
/// </summary>
public class RidgeRegression
{
    private double[]? _weights;
    private double _intercept;
    private double _alpha;

    /// <summary>
    /// 初始化岭回归模型
    /// </summary>
    /// <param name="alpha">正则化强度，值越大正则化越强</param>
    public RidgeRegression(double alpha = 1.0)
    {
        _alpha = alpha;
    }

    /// <summary>
    /// 训练岭回归模型
    /// </summary>
    public void Fit(double[,] X, double[] y)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);

        // 添加偏置项
        double[,] XWithBias = new double[n, m + 1];
        for (int i = 0; i < n; i++)
        {
            XWithBias[i, 0] = 1.0;
            for (int j = 0; j < m; j++)
                XWithBias[i, j + 1] = X[i, j];
        }

        // 岭回归正规方程: w = (X^T * X + α*I)^(-1) * X^T * y
        double[,] XTranspose = Transpose(XWithBias);
        double[,] XTX = MatrixMultiply(XTranspose, XWithBias);

        // 添加正则化项 α*I（不对截距项正则化）
        for (int i = 1; i < XTX.GetLength(0); i++)
            XTX[i, i] += _alpha;

        double[,] XTXInverse = MatrixInverse(XTX);
        double[,] XTXInverseXT = MatrixMultiply(XTXInverse, XTranspose);
        double[] weights = MatrixVectorMultiply(XTXInverseXT, y);

        _intercept = weights[0];
        _weights = new double[m];
        Array.Copy(weights, 1, _weights, 0, m);
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

    #region 矩阵运算
    private static double[,] Transpose(double[,] matrix)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        double[,] result = new double[cols, rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j, i] = matrix[i, j];
        return result;
    }

    private static double[,] MatrixMultiply(double[,] a, double[,] b)
    {
        int aRows = a.GetLength(0), aCols = a.GetLength(1), bCols = b.GetLength(1);
        double[,] result = new double[aRows, bCols];
        for (int i = 0; i < aRows; i++)
            for (int j = 0; j < bCols; j++)
                for (int k = 0; k < aCols; k++)
                    result[i, j] += a[i, k] * b[k, j];
        return result;
    }

    private static double[] MatrixVectorMultiply(double[,] matrix, double[] vector)
    {
        int rows = matrix.GetLength(0), cols = matrix.GetLength(1);
        double[] result = new double[rows];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[i] += matrix[i, j] * vector[j];
        return result;
    }

    private static double[,] MatrixInverse(double[,] matrix)
    {
        int n = matrix.GetLength(0);
        double[,] result = new double[n, n];
        double[,] temp = (double[,])matrix.Clone();

        for (int i = 0; i < n; i++)
            result[i, i] = 1.0;

        for (int i = 0; i < n; i++)
        {
            double pivot = temp[i, i];
            for (int j = 0; j < n; j++)
            {
                temp[i, j] /= pivot;
                result[i, j] /= pivot;
            }

            for (int k = 0; k < n; k++)
            {
                if (k != i)
                {
                    double factor = temp[k, i];
                    for (int j = 0; j < n; j++)
                    {
                        temp[k, j] -= factor * temp[i, j];
                        result[k, j] -= factor * result[i, j];
                    }
                }
            }
        }
        return result;
    }
    #endregion
}
