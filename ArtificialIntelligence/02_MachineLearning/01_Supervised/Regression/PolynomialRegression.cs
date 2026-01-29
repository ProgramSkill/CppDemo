namespace ArtificialIntelligence.MachineLearning.Supervised.Regression;

/// <summary>
/// 多项式回归
/// 通过特征的多项式组合来拟合非线性关系
/// 例如: y = w0 + w1*x + w2*x² + w3*x³
/// </summary>
public class PolynomialRegression
{
    private LinearRegression _linearModel;
    private int _degree;

    /// <summary>
    /// 初始化多项式回归
    /// </summary>
    /// <param name="degree">多项式的阶数</param>
    public PolynomialRegression(int degree = 2)
    {
        if (degree < 1)
            throw new ArgumentException("阶数必须大于等于1");

        _degree = degree;
        _linearModel = new LinearRegression();
    }

    /// <summary>
    /// 训练模型
    /// </summary>
    public void Fit(double[,] X, double[] y)
    {
        double[,] XPoly = TransformFeatures(X);
        _linearModel.Fit(XPoly, y);
    }

    /// <summary>
    /// 预测
    /// </summary>
    public double[] Predict(double[,] X)
    {
        double[,] XPoly = TransformFeatures(X);
        return _linearModel.Predict(XPoly);
    }

    /// <summary>
    /// 将特征转换为多项式特征
    /// </summary>
    private double[,] TransformFeatures(double[,] X)
    {
        int n = X.GetLength(0);
        int m = X.GetLength(1);

        // 计算多项式特征的数量
        int polyFeatures = m * _degree;
        double[,] XPoly = new double[n, polyFeatures];

        for (int i = 0; i < n; i++)
        {
            int featureIndex = 0;
            for (int j = 0; j < m; j++)
            {
                for (int d = 1; d <= _degree; d++)
                {
                    XPoly[i, featureIndex++] = Math.Pow(X[i, j], d);
                }
            }
        }

        return XPoly;
    }
}
