import numpy
import pandas
from scipy import stats
from statsmodels.api import OLS
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from statsmodels.tsa.stattools import acf, adfuller
from statsmodels.graphics.gofplots import qqplot
from statsmodels.graphics.tsaplots import plot_acf

zeros = pandas.read_excel('zeros.xlsx')
vix = zeros['VIX'].values
zeros = zeros.drop(['Month', 'VIX', '3M'], axis = 1)
N = len(vix)
print('Number of data points = ', N)
lvix = numpy.log(vix)
print('ADF test p = ', adfuller(lvix)[1])
VIXAR = stats.linregress(lvix[:-1], numpy.diff(lvix))
print('Autoregression VIX')
print(VIXAR)
vixres = numpy.array([lvix[k+1] - lvix[k] * (VIXAR.slope + 1)- VIXAR.intercept for k in range(N-1)])
print('Skewness and kurtosis = ', round(stats.skew(vixres), 2), round(stats.kurtosis(vixres, fisher = False), 2))
print(round(acf(vixres, qstat = True, nlags = 10)[2][-1], 2))
print(round(acf(abs(vixres), qstat = True, nlags = 10)[2][-1], 2))
# qqplot(vixres, line = 's')
# plt.show()
# plot_acf(vixres)
# plt.show()
# plot_acf(abs(vixres))
# plt.show()

def plots(data, label):
    plot_acf(data, zero = False)
    plt.title(label + '\n ACF for Original Values')
    plt.show()
    plot_acf(abs(data), zero = False)
    plt.title(label + '\n ACF for Absolute Values')
    plt.show()
    qqplot(data, line = 's')
    plt.title(label + '\n Quantile-Quantile Plot vs Normal')
    plt.show()
    
def analysis(data, key):
    print(key)
    print('Skewness and kurtosis = ', round(stats.skew(data), 2), round(stats.kurtosis(data, fisher = False), 2))
    print('ACF for original and absolute values L1 norm = ', round(sum(abs(acf(data, nlags = 5)[1:])), 2), round(sum(abs(acf(abs(data), nlags = 5)[1:])), 2))

pca = PCA(n_components=3)
pca.fit(zeros)
print('Explained variance ratios % for first three PC:', [round(100*pca.explained_variance_ratio_[n], 2) for n in range(3)])
components = pca.transform(zeros)
for n in range(3):
    plt.plot(components[:, n], label = 'PC'+str(n+1))
plt.legend(loc = 'upper right')
plt.xlabel('Time in Months')
plt.title('Time series of the first three PC, 1990-2024')
plt.show()
print('First loading vector')
print(pca.components_[0])
print('Second loading vector')
print(pca.components_[1])
print('Third loading vector')
print(pca.components_[2])
for n in range(3):
    plt.plot(pca.components_[n], label = 'PC'+str(n+1))
plt.legend(loc = 'upper right')
plt.title('Loading factors for the first three PC')
plt.xlabel('Maturity in years')
plt.show()

for n in range(3):
    print('PC'+str(n+1), '\n')
    series = components[:, n]
    print('ADF test p = ', adfuller(series)[1])
    reg = stats.linregress(series[:-1], numpy.diff(series))
    print(reg)
    res = numpy.array([numpy.diff(series)[k] - reg.slope * series[k] - reg.intercept for k in range(N-1)])
    analysis(res, 'univariate original residuals for PC'+str(n+1))
    plots(res, 'univariate original residuals for PC'+str(n+1))
    print('\n')
    nres = res/vix[1:]
    analysis(nres, 'univariate normalized residuals for PC'+str(n+1))
    plots(nres, 'univariate normalized residuals for PC'+str(n+1))

level = components[:, 0]
slope = components[:, 1]
curvature = components[:, 2]
print('bivariate model')
normDF = pandas.DataFrame({'const' : 1/vix[1:], 'vix': 1, 'Level': level[:-1]/vix[1:], 'Slope' : slope[:-1]/vix[1:]})
origDF = pandas.DataFrame({'const' : 1, 'vix' : vix[1:], 'Level': level[:-1], 'Slope' : slope[:-1]})
levelReg = OLS(numpy.diff(level), origDF).fit()
print(levelReg.summary())
slopeReg = OLS(numpy.diff(slope)/vix[1:], normDF).fit()
print(slopeReg.summary())
levelResid = levelReg.resid
analysis(levelResid, 'level residuals = ')
slopeResid = slopeReg.resid
analysis(slopeResid, 'slope residuals = ')
covMatrix = numpy.corrcoef([vixres, levelResid, slopeResid])
print('Correlation matrix between residuals of AR(1) ln VIX, Level, Slope')
print(numpy.matrix.round(covMatrix, 2))
print('stdev of vix residuals = ', numpy.std(vixres))
print('stdev of level residuals = ', numpy.std(levelResid))
print('stdev of slope residuals = ', numpy.std(slopeResid))
print('trivariate model')
normDF = pandas.DataFrame({'const' : 1/vix[1:], 'vix': 1, 'Level': level[:-1]/vix[1:], 'Slope' : slope[:-1]/vix[1:], 'Curvature' : curvature[:-1]/vix[1:]})
origDF = pandas.DataFrame({'const' : 1, 'vix' : vix[1:], 'Level': level[:-1], 'Slope' : slope[:-1], 'Curvature' : curvature[:-1]})
levelReg = OLS(numpy.diff(level), origDF).fit()
print(levelReg.summary())
slopeReg = OLS(numpy.diff(slope)/vix[1:], normDF).fit()
print(slopeReg.summary())
curvatureReg = OLS(numpy.diff(curvature), origDF).fit()
print(curvatureReg.summary())
levelResid = levelReg.resid
analysis(levelResid, 'level residuals = ')
slopeResid = slopeReg.resid
analysis(slopeResid, 'slope residuals = ')
curvatureResid = curvatureReg.resid
analysis(curvatureResid, 'curvature residuals = ')
covMatrix = numpy.corrcoef([vixres, levelResid, slopeResid, curvatureResid])
print('Correlation matrix between residuals of AR(1) ln VIX, Level, Slope, Curvature')
print(numpy.matrix.round(covMatrix, 2))
print('stdev of vix residuals = ', numpy.std(vixres))
print('stdev of level residuals = ', numpy.std(levelResid))
print('stdev of slope residuals = ', numpy.std(slopeResid))
print('stdev of curvature residuals = ', numpy.std(curvatureResid))