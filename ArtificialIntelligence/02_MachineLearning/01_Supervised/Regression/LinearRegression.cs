namespace ArtificialIntelligence.MachineLearning.Supervised.Regression;

/// <summary>
/// 线性回归实现
/// 使用最小二乘法拟合线性模型: y = w0 + w1*x1 + w2*x2 + ... + wn*xn
/// </summary>
public class LinearRegression
{
    private double[]? _weights;
    private double _intercept;

    /// <summary>
    /// 训练线性回归模型
    /// </summary>
    /// <param name="X">特征矩阵 [样本数 x 特征数]</param>
    /// <param name="y">目标值数组</param>
    public void Fit(double[,] X, double[] y)
    {
        int n = X.GetLength(0); // 样本数
        int m = X.GetLength(1); // 特征数

        // 添加偏置项（截距）
        double[,] XWithBias = new double[n, m + 1];
        for (int i = 0; i < n; i++)
        {
            XWithBias[i, 0] = 1.0; // 偏置项
            for (int j = 0; j < m; j++)
            {
                XWithBias[i, j + 1] = X[i, j];
            }
        }

        // 使用正规方程求解: w = (X^T * X)^(-1) * X^T * y
        double[,] XTranspose = Transpose(XWithBias);
        double[,] XTX = MatrixMultiply(XTranspose, XWithBias);
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
            throw new InvalidOperationException("模型未训练，请先调用Fit方法");

        int n = X.GetLength(0);
        int m = X.GetLength(1);
        double[] predictions = new double[n];

        for (int i = 0; i < n; i++)
        {
            predictions[i] = _intercept;
            for (int j = 0; j < m; j++)
            {
                predictions[i] += _weights[j] * X[i, j];
            }
        }

        return predictions;
    }

    #region 矩阵运算辅助方法

    private static double[,] Transpose(double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
        double[,] result = new double[cols, rows];

        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                result[j, i] = matrix[i, j];

        return result;
    }

    private static double[,] MatrixMultiply(double[,] a, double[,] b)
    {
        int aRows = a.GetLength(0);
        int aCols = a.GetLength(1);
        int bCols = b.GetLength(1);
        double[,] result = new double[aRows, bCols];

        for (int i = 0; i < aRows; i++)
            for (int j = 0; j < bCols; j++)
                for (int k = 0; k < aCols; k++)
                    result[i, j] += a[i, k] * b[k, j];

        return result;
    }

    private static double[] MatrixVectorMultiply(double[,] matrix, double[] vector)
    {
        int rows = matrix.GetLength(0);
        int cols = matrix.GetLength(1);
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

        // 创建单位矩阵
        for (int i = 0; i < n; i++)
            result[i, i] = 1.0;

        // 高斯-约旦消元法
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
