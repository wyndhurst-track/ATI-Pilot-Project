import os
import sys
sys.path.append('../../../')
import json
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True
from sklearn.neighbors import KNeighborsClassifier

# My libraries
from Utilities.DataUtils import DataUtils

"""
TEMPORARY: because of a bug with not resetting some global variables in utils.py between
folds, logs of accuracies and loss were concatenated and need to be segragated
"""

def plotAccuracyVsFolds(plot_error_bars=True):
	# Data (average accuracy vs folds)
	TL = np.array([[99.24,10],[98.82,17],[97.98,25],[97.84667,33],[95.865,50],[93.34,60],[84.07,70],[81.65,80],[71.57,90]])
	RTL = np.array([[99.155,10],[98.725,17],[98.6875,25],[98.59,33],[96.775,50],[93.35,60],[94.96,70],[89.11,80],[87.7,90]])
	STL = np.array([[99.6371,10],[99.56317,17],[99.14,25],[98.857,33],[97.58,50],[97.782,60],[94.556,70],[89.314,80],[86.895,90]])
	SRTL = np.array([[99.67742,10],[99.63,17],[98.53831,25],[98.99194,33],[98.18548,50],[97.37903,60],[95.56452,70],[89.51613,80],[94.47581,90]])
	softmax = np.array([[89.798,10],[83.165,17],[74.85,25],[66.53333,33],[49.9,50],[36.69,60],[25.6,70],[13.1,80],[7.86,90]])

	# Min and maximum values for each method (range)
	range_TL = np.array([[0.24,0.64,2.02,0.88,0.51,0,0,0,0],[0.36,0.37,1.01,0.74,0.51,0,0,0,0]])
	range_RTL = np.array([[0.97,0.95,0.71,0.2,0.61,0,0,0,0],[0.45,0.68,0.5,0.2,0.61,0,0,0,0]])
	range_STL = np.array([[0.24,0.17,0.76,0.67,0.2,0,0,0,0],[0.16,0.24,0.45,0.54,0.2,0,0,0,0]])
	range_SRTL = np.array([[0.69,0.24,1.16,0,0.6,0,0,0,0],[0.12,0.17,0.86,0,0.6,0,0,0,0]])
	range_SM = np.array([[7.95,4.54,9.12,6.85,0.3,0,0,0,0],[6.77,2.93,7.61,8.87,0.3,0,0,0,0]])

	fig, ax = plt.subplots()

	if plot_error_bars:
		ax.errorbar(TL[:,1], TL[:,0], yerr=range_TL, label="TL")
		ax.errorbar(RTL[:,1], RTL[:,0], yerr=range_RTL, label="RTL")
		ax.errorbar(STL[:,1], STL[:,0], yerr=range_STL, label="SoftMax+TL")
		ax.errorbar(SRTL[:,1], SRTL[:,0], yerr=range_SRTL, label="Softmax+RTL") 
		ax.errorbar(softmax[:,1], softmax[:,0], yerr=range_SM, label="SoftMax (closed-set)")
	else:
		ax.plot(TL[:,1], TL[:,0], label="TL", marker="^")
		ax.plot(RTL[:,1], RTL[:,0], label="RTL", marker="o")
		ax.plot(STL[:,1], STL[:,0], label="SoftMax+TL", marker="*")
		ax.plot(SRTL[:,1], SRTL[:,0], label="Softmax+RTL", marker="+") 
		ax.plot(softmax[:,1], softmax[:,0], label="SoftMax (closed-set)", marker="s")

	ax.legend(loc='center upper')

	ax.set_xlabel('Openness (%)')
	ax.set_ylabel('Average accuracy (%)')
	ax.set_xlim((9,90))
	ax.set_ylim((0,100))

	plt.grid(True)
	plt.tight_layout()
	plt.savefig("acc-vs-open.pdf")
	plt.show()

def plotAccuracyVsUnknownRatio():
	unknown_ratios = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.33, 0.25, 0.17, 0.1]) * 100

	# REP 0
	valid_TL_0 = np.array([79.60088692, 87.80487805, 82.92682927, 85.58758315, 91.1308204, 93.12638581, 92.23946785, 92.90465632, 93.79157428])
	valid_RTL_0 = np.array([83.81374723, 90.90909091, 90.90909091, 92.01773836, 94.23503326, 96.00886918, 94.67849224, 95.5654102, 94.67849224])
	valid_STL_0 = np.array([86.03104213, 92.01773836, 88.9135255, 93.12638581, 94.45676275, 96.45232816, 96.00886918, 96.00886918, 96.00886918])
	valid_SRTL_0 = np.array([85.14412417, 91.79600887, 89.57871397, 92.90465632, 94.90022173, 95.5654102, 95.78713969, 96.00886918, 95.78713969])
	valid_closed_0 = np.array([16.88888889, 28.44444444, 30.44444444, 36.88888889, 54, 58, 65.33333333, 82, 85.33333333])
	test_TL_0 = np.array([80.91106291, 84.164859, 78.30802603, 83.94793926, 86.3340564, 88.06941432, 85.68329718, 82.21258134, 79.17570499])
	test_RTL_0 = np.array([88.93709328, 87.63557484, 91.54013015, 93.05856833, 93.27548807, 90.23861171, 92.40780911, 85.46637744, 82.64642082])
	test_STL_0 = np.array([88.72017354, 90.45553145, 92.19088937, 93.92624729, 96.09544469, 96.96312364, 94.79392625, 91.10629067, 89.58785249])
	test_SRTL_0 = np.array([88.93709328, 91.75704989, 92.84164859, 94.57700651, 94.36008677, 94.57700651, 92.40780911, 86.11713666, 90.02169197])
	test_closed_0 = np.array([11.30434783, 27.39130435, 34.34782609, 46.30434783, 48.69565217, 66.95652174, 68.04347826, 65, 65.65217391])

	# REP 1
	valid_TL_1 = np.array([77.60532151, 81.15299335, 87.80487805, 89.80044346, 91.1308204, 90.68736142, 92.68292683, 94.23503326, 91.57427938])
	valid_RTL_1 = np.array([83.59201774, 90.02217295, 94.23503326, 94.23503326, 94.90022173, 94.45676275, 95.5654102, 96.89578714, 95.34368071])
	valid_STL_1 = np.array([83.14855876, 86.91796009, 93.3481153, 94.23503326, 94.23503326, 94.23503326, 96.00886918, 97.56097561, 96.45232816])
	valid_SRTL_1 = np.array([85.80931264, 89.80044346, 92.68292683, 94.45676275, 95.78713969, 95.12195122, 95.5654102, 97.56097561, 96.23059867])
	valid_closed_1 = np.array([15.55555556, 20, 27.55555556, 49.33333333, 45.33333333, 71.33333333, 73.33333333, 81.77777778, 90.22222222])
	test_TL_1 = np.array([76.35574837, 72.6681128, 87.20173536, 78.52494577, 90.45553145, 79.39262473, 78.95878525, 84.81561822, 79.82646421])
	test_RTL_1 = np.array([86.3340564, 87.63557484, 93.05856833, 86.11713666, 95.66160521, 86.55097614, 92.62472885, 88.93709328, 85.03253796])
	test_STL_1 = np.array([86.76789588, 86.55097614, 93.70932755, 91.54013015, 94.14316703, 92.84164859, 95.22776573, 93.92624729, 88.06941432])
	test_SRTL_1 = np.array([86.76789588, 88.72017354, 92.40780911, 90.88937093, 95.01084599, 93.27548807, 92.62472885, 93.49240781, 89.15401302])
	test_closed_1 = np.array([11.08695652, 23.04347826, 31.52173913, 38.91304348, 51.73913043, 55.86956522, 68.04347826, 71.30434783, 66.30434783])

	# REP 2
	valid_TL_2 = np.array([76.94013304, 86.03104213, 89.80044346, 88.47006652, 91.1308204, 92.90465632, 94.01330377, 92.90465632, 92.68292683])
	valid_RTL_2 = np.array([76.71840355, 92.23946785, 93.12638581, 93.12638581, 95.34368071, 96.45232816, 96.00886918, 92.90465632, 95.78713969])
	valid_STL_2 = np.array([76.27494457, 92.68292683, 94.90022173, 91.1308204, 95.12195122, 97.56097561, 97.11751663, 96.89578714, 96.45232816])
	valid_SRTL_2 = np.array([78.27050998, 91.79600887, 94.01330377, 92.46119734, 95.78713969, 96.00886918, 96.67405765, 96.23059867, 96.89578714])
	valid_closed_2 = np.array([7.333333333, 27.11111111, 34.88888889, 36.22222222, 54.88888889, 63.33333333, 78.22222222, 82.44444444, 86.22222222])
	test_TL_2 = np.array([79.82646421, 85.2494577, 90.45553145, 85.68329718, 79.60954447, 89.37093275, 84.164859, 83.0802603, 78.52494577])
	test_RTL_2 = np.array([78.09110629, 93.92624729, 93.05856833, 92.19088937, 92.40780911, 93.49240781, 89.15401302, 81.77874187, 86.11713666])
	test_STL_2 = np.array([80.47722343, 91.32321041, 91.32321041, 92.62472885, 91.32321041, 95.87852495, 93.27548807, 91.75704989, 91.32321041])
	test_SRTL_2 = np.array([83.51409978, 90.88937093, 91.32321041, 93.27548807, 95.66160521, 97.18004338, 93.49240781, 91.54013015, 91.10629067])
	test_closed_2 = np.array([14.34782609, 22.17391304, 31.73913043, 46.73913043, 47.17391304, 67.60869565, 64.56521739, 68.04347826, 66.73913043])

	# REP 3
	valid_TL_3 = np.array([72.06208426, 87.36141907, 80.48780488, 88.02660754, 91.35254989, 92.68292683, 93.3481153, 95.34368071, 92.46119734])
	valid_RTL_3 = np.array([84.25720621, 90.90909091, 91.79600887, 92.46119734, 95.78713969, 94.45676275, 96.45232816, 97.7827051, 96.45232816])
	valid_STL_3 = np.array([87.80487805, 89.57871397, 90.90909091, 92.23946785, 94.90022173, 94.90022173, 97.11751663, 98.66962306, 98.22616408])
	valid_SRTL_3 = np.array([85.58758315, 89.13525499, 90.90909091, 93.12638581, 95.12195122, 95.12195122, 96.67405765, 98.00443459, 97.7827051])
	valid_closed_3 = np.array([12.88888889, 21.77777778, 24.88888889, 38.88888889, 44.44444444, 63.33333333, 71.11111111, 78.22222222, 86.22222222])
	test_TL_3 = np.array([72.88503254, 88.28633406, 78.95878525, 80.26030369, 86.3340564, 87.85249458, 86.11713666, 84.59869848, 84.164859])
	test_RTL_3 = np.array([84.164859, 93.49240781, 89.37093275, 93.05856833, 93.05856833, 93.05856833, 94.14316703, 94.57700651, 93.49240781])
	test_STL_3 = np.array([89.58785249, 91.10629067, 90.45553145, 93.92624729, 94.36008677, 97.18004338, 98.04772234, 99.13232104, 97.18004338])
	test_SRTL_3 = np.array([89.80477223, 87.63557484, 92.40780911, 92.19088937, 94.79392625, 95.22776573, 97.8308026, 96.09544469, 97.18004338])
	test_closed_3 = np.array([11.30434783, 23.91304348, 34.56521739, 42.60869565, 49.34782609, 67.82608696, 71.30434783, 80.43478261, 73.47826087])

	# REP 4
	valid_TL_4 = np.array([74.50110865, 81.59645233, 85.14412417, 91.79600887, 89.80044346, 94.45676275, 95.12195122, 92.01773836, 94.01330377])
	valid_RTL_4 = np.array([81.81818182, 90.68736142, 86.91796009, 95.34368071, 95.34368071, 96.00886918, 96.67405765, 95.12195122, 97.33924612])
	valid_STL_4 = np.array([84.03547672, 90.24390244, 92.01773836, 94.67849224, 94.01330377, 97.56097561, 97.7827051, 95.12195122, 98.00443459])
	valid_SRTL_4 = np.array([83.81374723, 89.80044346, 92.01773836, 94.45676275, 95.12195122, 96.23059867, 97.11751663, 95.78713969, 97.56097561])
	valid_closed_4 = np.array([11.77777778, 29.77777778, 33.55555556, 44.44444444, 44, 65.33333333, 69.33333333, 74.66666667, 84])
	test_TL_4 = np.array([67.02819957, 83.94793926, 78.09110629, 87.20173536, 86.3340564, 87.63557484, 87.63557484, 79.60954447, 83.73101952])
	test_RTL_4 = np.array([82.64642082, 88.93709328, 83.0802603, 91.54013015, 90.02169197, 93.05856833, 92.40780911, 83.0802603, 91.32321041])
	test_STL_4 = np.array([86.76789588, 92.84164859, 88.72017354, 94.36008677, 92.40780911, 94.36008677, 92.62472885, 90.23861171, 94.36008677])
	test_SRTL_4 = np.array([83.94793926, 91.54013015, 90.45553145, 94.14316703, 91.75704989, 97.39696312, 93.27548807, 88.28633406, 96.31236443])
	test_closed_4 = np.array([14.7826087, 21.08695652, 35.86956522, 40.86956522, 53.04347826, 60.43478261, 72.60869565, 66.73913043, 78.69565217])

	# REP 5
	valid_TL_5 = np.array([75.16629712, 84.70066519, 80.70953437, 91.1308204, 91.57427938, 92.90465632, 93.12638581, 93.79157428, 93.79157428])
	valid_RTL_5 = np.array([82.2616408, 93.3481153, 91.79600887, 94.23503326, 95.5654102, 94.90022173, 95.34368071, 95.34368071, 96.23059867])
	valid_STL_5 = np.array([82.92682927, 92.46119734, 92.90465632, 95.5654102, 95.12195122, 96.00886918, 95.78713969, 95.5654102, 96.67405765])
	valid_SRTL_5 = np.array([83.14855876, 89.80044346, 90.46563193, 94.90022173, 95.5654102, 96.23059867, 96.67405765, 95.78713969, 96.45232816])
	valid_closed_5 = np.array([9.555555556, 25.33333333, 32.22222222, 37.33333333, 54.88888889, 76.22222222, 78.66666667, 86.22222222, 83.55555556])
	test_TL_5 = np.array([81.34490239, 78.52494577, 74.8373102, 83.73101952, 78.74186551, 83.29718004, 79.60954447, 79.17570499, 80.69414317])
	test_RTL_5 = np.array([83.94793926, 89.15401302, 91.75704989, 92.40780911, 88.28633406, 90.02169197, 84.38177874, 83.0802603, 91.75704989])
	test_STL_5 = np.array([89.37093275, 91.10629067, 91.10629067, 95.44468547, 90.88937093, 92.84164859, 94.79392625, 86.76789588, 95.22776573])
	test_SRTL_5 = np.array([89.37093275, 88.28633406, 91.32321041, 93.92624729, 93.70932755, 93.70932755, 95.01084599, 86.3340564, 93.27548807])
	test_closed_5 = np.array([14.56521739, 22.60869565, 27.82608696, 43.47826087, 40.65217391, 66.30434783, 64.13043478, 66.08695652, 70.43478261])

	# REP 6
	valid_TL_6 = np.array([79.37915743, 86.91796009, 92.01773836, 91.57427938, 88.9135255, 94.67849224, 94.01330377, 91.79600887, 94.45676275])
	valid_RTL_6 = np.array([88.69179601, 92.23946785, 94.90022173, 92.90465632, 95.5654102, 96.23059867, 97.11751663, 93.56984479, 96.23059867])
	valid_STL_6 = np.array([85.14412417, 91.57427938, 94.67849224, 96.23059867, 92.68292683, 96.67405765, 97.33924612, 96.45232816, 97.11751663])
	valid_SRTL_6 = np.array([82.70509978, 92.46119734, 94.67849224, 97.11751663, 92.68292683, 96.23059867, 96.45232816, 96.00886918, 96.89578714])
	valid_closed_6 = np.array([15.55555556, 24.44444444, 38, 49.77777778, 51.11111111, 65.11111111, 73.11111111, 84.22222222, 83.33333333])
	test_TL_6 = np.array([80.47722343, 79.60954447, 89.80477223, 81.12798265, 83.73101952, 83.94793926, 80.47722343, 78.30802603, 81.34490239])
	test_RTL_6 = np.array([86.76789588, 88.28633406, 94.36008677, 84.81561822, 94.36008677, 89.37093275, 89.58785249, 86.11713666, 91.97396963])
	test_STL_6 = np.array([88.28633406, 90.02169197, 94.79392625, 92.40780911, 94.14316703, 92.19088937, 94.14316703, 90.88937093, 94.57700651])
	test_SRTL_6 = np.array([88.06941432, 90.23861171, 94.57700651, 92.84164859, 94.79392625, 91.32321041, 93.05856833, 89.15401302, 96.7462039])
	test_closed_6 = np.array([13.69565217, 23.91304348, 33.91304348, 41.30434783, 54.7826087, 60.65217391, 68.26086957, 68.47826087, 79.7826087])

	# REP 7
	valid_TL_7 = np.array([75.6097561, 82.03991131, 88.69179601, 88.24833703, 93.12638581, 92.68292683, 95.12195122, 92.90465632, 92.01773836])
	valid_RTL_7 = np.array([81.81818182, 90.90909091, 91.1308204, 94.67849224, 96.67405765, 94.45676275, 97.33924612, 96.00886918, 95.5654102])
	valid_STL_7 = np.array([83.37028825, 91.35254989, 94.01330377, 94.90022173, 94.90022173, 96.00886918, 98.22616408, 96.89578714, 96.23059867])
	valid_SRTL_7 = np.array([83.37028825, 93.12638581, 94.23503326, 96.89578714, 96.23059867, 94.90022173, 98.44789357, 97.11751663, 95.78713969])
	valid_closed_7 = np.array([11.55555556, 20.66666667, 35.77777778, 51.77777778, 50.88888889, 65.55555556, 73.77777778, 84, 90.44444444])
	test_TL_7 = np.array([80.04338395, 83.73101952, 81.12798265, 79.60954447, 85.46637744, 85.46637744, 86.55097614, 82.64642082, 77.00650759])
	test_RTL_7 = np.array([86.11713666, 91.10629067, 83.94793926, 92.62472885, 92.84164859, 93.05856833, 95.87852495, 85.2494577, 87.63557484])
	test_STL_7 = np.array([87.63557484, 91.54013015, 90.02169197, 94.57700651, 92.62472885, 93.27548807, 96.7462039, 94.79392625, 92.40780911])
	test_SRTL_7 = np.array([87.4186551, 91.97396963, 93.05856833, 91.10629067, 95.44468547, 94.36008677, 95.66160521, 94.57700651, 91.54013015])
	test_closed_7 = np.array([13.69565217, 24.7826087, 32.39130435, 45.43478261, 51.30434783, 71.30434783, 74.7826087, 67.82608696, 65.2173913])

	# REP 8
	valid_TL_8 = np.array([82.92682927, 88.69179601, 88.9135255, 90.68736142, 90.68736142, 94.90022173, 95.5654102, 93.56984479, 93.56984479])
	valid_RTL_8 = np.array([89.57871397, 92.46119734, 93.12638581, 94.67849224, 94.01330377, 96.45232816, 96.23059867, 96.45232816, 95.5654102])
	valid_STL_8 = np.array([89.35698448, 92.90465632, 92.01773836, 95.78713969, 94.67849224, 96.45232816, 98.00443459, 97.11751663, 96.45232816])
	valid_SRTL_8 = np.array([85.14412417, 90.90909091, 90.90909091, 95.78713969, 93.12638581, 95.78713969, 97.33924612, 97.11751663, 96.00886918])
	valid_closed_8 = np.array([13.55555556, 23.55555556, 29.77777778, 40.22222222, 49.33333333, 61.77777778, 74.44444444, 80.88888889, 90.66666667])
	test_TL_8 = np.array([83.29718004, 86.98481562, 90.45553145, 83.94793926, 83.29718004, 84.81561822, 88.06941432, 78.30802603, 80.91106291])
	test_RTL_8 = np.array([91.75704989, 85.68329718, 90.45553145, 91.54013015, 89.58785249, 92.62472885, 92.62472885, 88.72017354, 86.76789588])
	test_STL_8 = np.array([89.15401302, 89.37093275, 92.19088937, 95.01084599, 92.84164859, 94.57700651, 98.04772234, 93.27548807, 87.85249458])
	test_SRTL_8 = np.array([88.5032538, 90.02169197, 92.62472885, 93.49240781, 90.23861171, 95.22776573, 97.61388286, 92.84164859, 88.06941432])
	test_closed_8 = np.array([10.65217391, 20.2173913, 31.95652174, 46.08695652, 51.08695652, 62.39130435, 67.39130435, 64.13043478, 73.26086957])

	# REP 9
	valid_TL_9 = np.array([81.15299335, 76.27494457, 91.35254989, 90, 90.24390244, 94.23503326, 95.5654102, 91.79600887, 91.35254989])
	valid_RTL_9 = np.array([82.92682927, 66.7405765, 93.12638581, 90.68736142, 95.78713969, 95.78713969, 97.33924612, 95.5654102, 96.23059867])
	valid_STL_9 = np.array([82.92682927, 90.24390244, 93.3481153, 95.78713969, 94.01330377, 96.45232816, 98.00443459, 96.67405765, 95.78713969])
	valid_SRTL_9 = np.array([83.37028825, 90.90909091, 91.79600887, 94.45676275, 92.46119734, 94.90022173, 98.22616408, 96.45232816, 96.00886918])
	valid_closed_9 = np.array([12.66666667, 20.44444444, 41.77777778, 39.77777778, 53.77777778, 69.55555556, 74.22222222, 83.55555556, 88.22222222])
	test_TL_9 = np.array([85.46637744, 73.31887202, 81.34490239, 79, 76.35574837, 82.42950108, 86.55097614, 80.04338395, 78.30802603])
	test_RTL_9 = np.array([85.68329718, 65.94360087, 92.62472885, 85.2494577, 90.45553145, 88.28633406, 94.14316703, 84.81561822, 86.98481562])
	test_STL_9 = np.array([91.10629067, 90.23861171, 93.49240781, 95.01084599, 90.02169197, 90.45553145, 98.26464208, 95.44468547, 90.88937093])
	test_SRTL_9 = np.array([89.15401302, 91.10629067, 93.92624729, 93.70932755, 91.75704989, 91.32321041, 97.39696312, 94.14316703, 91.10629067])
	test_closed_9 = np.array([13.26086957, 26.30434783, 34.56521739, 37.39130435, 47.39130435, 61.73913043, 68.47826087, 63.47826087, 70.65217391])

	# Gathered
	valid_TL_reps = [valid_TL_0, valid_TL_1, valid_TL_2, valid_TL_3, valid_TL_4, valid_TL_5, valid_TL_6, valid_TL_7, valid_TL_8, valid_TL_9]
	valid_RTL_reps = [valid_RTL_0, valid_RTL_1, valid_RTL_2, valid_RTL_3, valid_RTL_4, valid_RTL_5, valid_RTL_6, valid_RTL_7, valid_RTL_8, valid_RTL_9]
	valid_STL_reps = [valid_STL_0, valid_STL_1, valid_STL_2, valid_STL_3, valid_STL_4, valid_STL_5, valid_STL_6, valid_STL_7, valid_STL_8, valid_STL_9]
	valid_SRTL_reps = [valid_SRTL_0, valid_SRTL_1, valid_SRTL_2, valid_SRTL_3, valid_SRTL_4, valid_SRTL_5, valid_SRTL_6, valid_SRTL_7, valid_SRTL_8, valid_SRTL_9]
	valid_closed_reps = [valid_closed_0, valid_closed_1, valid_closed_2, valid_closed_3, valid_closed_4, valid_closed_5, valid_closed_6, valid_closed_7, valid_closed_8, valid_closed_9]

	test_TL_reps = [test_TL_0, test_TL_1, test_TL_2, test_TL_3, test_TL_4, test_TL_5, test_TL_6, test_TL_7, test_TL_8, test_TL_9]
	test_RTL_reps = [test_RTL_0, test_RTL_1, test_RTL_2, test_RTL_3, test_RTL_4, test_RTL_5, test_RTL_6, test_RTL_7, test_RTL_8, test_RTL_9]
	test_STL_reps = [test_STL_0, test_STL_1, test_STL_2, test_STL_3, test_STL_4, test_STL_5, test_STL_6, test_STL_7, test_STL_8, test_STL_9]
	test_SRTL_reps = [test_SRTL_0, test_SRTL_1, test_SRTL_2, test_SRTL_3, test_SRTL_4, test_SRTL_5, test_SRTL_6, test_SRTL_7, test_SRTL_8, test_SRTL_9]
	test_closed_reps = [test_closed_0, test_closed_1, test_closed_2, test_closed_3, test_closed_4, test_closed_5, test_closed_6, test_closed_7, test_closed_8, test_closed_9]

	# AVERAGE
	valid_TL = np.array([77.49445676, 84.25720621, 86.78492239, 89.48016753, 90.90909091, 93.32594235, 94.07982262, 93.12638581, 92.97117517])
	valid_RTL = np.array([83.54767184, 89.04656319, 92.10643016, 93.4368071, 95.32150776, 95.5210643, 96.27494457, 95.5210643, 95.94235033])
	valid_STL = np.array([84.10199557, 90.99778271, 92.70509978, 94.36807095, 94.41241685, 96.23059867, 97.13968958, 96.6962306, 96.7405765])
	valid_SRTL = np.array([83.63636364, 90.95343681, 92.1286031, 94.65631929, 94.67849224, 95.6097561, 96.89578714, 96.6075388, 96.54101996])
	valid_closed = np.array([12.73333333, 24.15555556, 32.88888889, 42.46666667, 50.26666667, 65.95555556, 73.15555556, 81.8, 86.82222222])
	test_TL = np.array([78.76355748, 81.64859002, 83.05856833, 82.67052302, 83.6659436, 85.22776573, 84.38177874, 81.27982646, 80.36876356])
	test_RTL = np.array([85.44468547, 87.18004338, 90.32537961, 90.26030369, 91.99566161, 90.97613883, 91.73535792, 86.18221258, 88.37310195])
	test_STL = np.array([87.78741866, 90.45553145, 91.80043384, 93.88286334, 92.88503254, 94.05639913, 95.59652928, 92.73318872, 92.14750542])
	test_SRTL = np.array([87.54880694, 90.21691974, 92.49457701, 93.01518438, 93.7527115, 94.36008677, 94.8373102, 91.25813449, 92.45119306])
	test_closed = np.array([12.86956522, 23.54347826, 32.86956522, 42.91304348, 49.52173913, 64.10869565, 68.76086957, 68.15217391, 71.02173913])

	colours = np.array(sns.color_palette("hls", 5))

	fig, axs = plt.subplots(1, 2, figsize=(10,5))

	lower_x_lim = 0

	alpha = 0.2

	for i in range(10):
		axs[0].plot(unknown_ratios, valid_TL_reps[i], color=np.insert(colours[0], 3, alpha))
		axs[0].plot(unknown_ratios, valid_RTL_reps[i], color=np.insert(colours[1], 3, alpha))
		axs[0].plot(unknown_ratios, valid_STL_reps[i], color=np.insert(colours[2], 3, alpha))
		axs[0].plot(unknown_ratios, valid_SRTL_reps[i], color=np.insert(colours[3], 3, alpha))
		axs[0].plot(unknown_ratios, valid_closed_reps[i], color=np.insert(colours[4], 3, alpha))

		axs[1].plot(unknown_ratios, test_TL_reps[i], color=np.insert(colours[0], 3, alpha))
		axs[1].plot(unknown_ratios, test_RTL_reps[i], color=np.insert(colours[1], 3, alpha))
		axs[1].plot(unknown_ratios, test_STL_reps[i], color=np.insert(colours[2], 3, alpha))
		axs[1].plot(unknown_ratios, test_SRTL_reps[i], color=np.insert(colours[3], 3, alpha))
		axs[1].plot(unknown_ratios, test_closed_reps[i], color=np.insert(colours[4], 3, alpha))

	# Validation
	axs[0].plot(unknown_ratios, valid_TL, label="TL", ls='--', color=colours[0])
	axs[0].plot(unknown_ratios, valid_RTL, label="RTL", ls='--', color=colours[1])
	axs[0].plot(unknown_ratios, valid_STL, label="Softmax+TL", ls='--', color=colours[2])
	axs[0].plot(unknown_ratios, valid_SRTL, label="Softmax+RTL", ls='--', color=colours[3])
	axs[0].plot(unknown_ratios, valid_closed, label="Closed set", ls='--', color=colours[4])

	axs[0].set_title('Validation')
	axs[0].legend(loc='lower left')
	axs[0].set_ylabel('Accuracy (%)')
	axs[0].set_xlabel('Openness (%)')
	axs[0].set_ylim((lower_x_lim,100))
	axs[0].set_xlim((np.min(unknown_ratios), np.max(unknown_ratios)))

	# Testing
	axs[1].plot(unknown_ratios, test_TL, label="TL", ls='--', color=colours[0])
	axs[1].plot(unknown_ratios, test_RTL, label="RTL", ls='--', color=colours[1])
	axs[1].plot(unknown_ratios, test_STL, label="Softmax+TL", ls='--', color=colours[2])
	axs[1].plot(unknown_ratios, test_SRTL, label="Softmax+RTL", ls='--', color=colours[3])
	axs[1].plot(unknown_ratios, test_closed, label="Closed set", ls='--', color=colours[4])

	axs[1].set_title('Testing')
	# axs[1].legend(loc='lower left')
	axs[1].set_xlabel('Openness (%)')
	axs[1].set_ylim((lower_x_lim,100))
	axs[1].set_xlim((np.min(unknown_ratios), np.max(unknown_ratios)))

	# plt.grid(True)
	plt.tight_layout()
	# plt.show()
	plt.savefig("acc-vs-open.pdf")

def plotAccuracyLoss(	fold, 
						accuracy_all, 
						accuracies_known, 
						accuracies_novel, 
						accuracy_steps,
						loss_sum,
						loss_steps			):
	fig, ax1 = plt.subplots()

	color1 = 'tab:blue'
	ax1.set_xlabel('Training steps')
	ax1.set_ylabel('Accuracy', color=color1)
	ax1.set_xlim((0, np.max(loss_steps)))
	ax1.set_ylim((60, 100))
	ax1.plot(accuracy_steps, accuracy_all, color=color1, label="Known+Unknown")
	ax1.plot(accuracy_steps, accuracies_known, color="cyan", label="Known")
	ax1.tick_params(axis='y', labelcolor=color1)

	ax2 = ax1.twinx()

	color2 = 'tab:orange'
	ax2.set_ylabel('Loss', color=color2)
	ax2.plot(loss_steps, loss_sum, color=color2, label="Loss")
	ax2.tick_params(axis='y', labelcolor=color2)
	ax2.set_ylim((0,6))

	print(f"Best accuracies for fold {fold}:")
	print(f"All: {np.max(accuracy_all)}")
	print(f"Known: {np.max(accuracies_known)}")
	print(f"Novel: {np.max(accuracies_novel)}")
	print(f"Max training steps (accuracy): {np.max(accuracy_steps)}")
	print(f"Max training steps (loss): {np.max(loss_steps)}")

	plt.legend(loc="center right")
	# plt.title(f"Fold {fold+1}")
	plt.tight_layout()
	plt.show()
	# plt.savefig(f"fold_{fold+1}.pdf")

	# Finish up
	plt.cla()
	plt.clf()
	plt.close()

# When using KNN to classify the space, where does the proportion of the error lie, with the known or unknown classes?
# And with which classes in particular?
def plotErrorOrigin():
	# Directories
	base_dir = "D:\\Work\\results\\CEiA\\SRTL"
	splits_base_dir = "D:\\OneDrive - University of Bristol\\Work\\1-PostDoc\\Data\\OpenCows2020\\identification\\images"

	# Load the splits files
	with open(os.path.join(splits_base_dir, "known_unknown_splits.json")) as handle:
		known_unknown_splits = json.load(handle)
	with open(os.path.join(splits_base_dir, "train_valid_test_splits.json")) as handle:
		train_valid_test_splits = json.load(handle)

	# Which repitition number are we looking over
	repitition = 0

	# The number of distinct classes
	num_classes = len(train_valid_test_splits.keys())

	# Retrieve all the folders at this directory
	folders = DataUtils.allFoldersAtDir(base_dir)

	# Dictionary storing results
	x_labels = []
	known = []
	unknown = []

	# Accumulate counts of incorrect labels and total test images
	labels_accum_incorrect = np.zeros(num_classes)
	labels_accum_images = np.zeros(num_classes)

	folder_dict = {'01':'90-10', '017':'83-17', '025':'75-25', '033':'67-33', '05':'50-50', '06':'40-60', '07':'30-70', '08':'20-80', '09':'10-90'}
	unknown_ratios = {'01':'0.1', '017':'0.17', '025':'0.25', '033':'0.33', '05':'0.5', '06':'0.6', '07':'0.7', '08':'0.8', '09':'0.9'}

	# Iterate through each folder
	for folder in folders:
		# Extract the folder name
		folder_name = os.path.basename(folder)
		x_labels.append(folder_dict[folder_name])

		# Create the path through the first fold
		npzfile_train = np.load(os.path.join(folder, f"rep_{repitition}/train_embeddings.npz"))
		npzfile_valid = np.load(os.path.join(folder, f"rep_{repitition}/valid_embeddings.npz"))
		npzfile_test = np.load(os.path.join(folder, f"rep_{repitition}/test_embeddings.npz"))

		# Extract the embeddings and labels
		X_train = np.concatenate((npzfile_train['embeddings'][1:], npzfile_valid['embeddings'][1:]))
		y_train = np.concatenate((npzfile_train['labels'][1:], npzfile_valid['labels'][1:]))
		X_test = npzfile_test['embeddings'][1:]
		y_test = npzfile_test['labels'][1:]

		# Create our classifier and use the training set to initialise it
		neigh = KNeighborsClassifier(n_neighbors=5, n_jobs=-4).fit(X_train, y_train)

		# Get the predictions based on this
		predictions = neigh.predict(X_test)

		# Which labels were incorrectly predicted?
		incorrect_labels = y_test[(predictions != y_test) == True]

		# What is the current unknown ratio
		unknown_ratio = unknown_ratios[folder_name]

		# Load the splits for this folder
		unknown_labels = [int(x) for x in known_unknown_splits[unknown_ratio][repitition]['unknown']]

		# Iterate over the incorrect labels to record some data
		known_incorrect = 0
		unknown_incorrect = 0
		for i in range(incorrect_labels.shape[0]):
			current_incorrect = incorrect_labels[i]

			if current_incorrect in unknown_labels: unknown_incorrect += 1
			else: known_incorrect += 1

			# Accumulate labels that were incorrectly predicted
			labels_accum_incorrect[int(current_incorrect)-1] += 1

		# Accumulate the total number of testing instances per class
		for i in range(npzfile_test['labels'][1:].shape[0]):
			current_label = npzfile_test['labels'][1:][i]
			labels_accum_images[int(current_label)-1] += 1

		# print(labels_accum_incorrect)
		# print(labels_accum_images)

		known_incorrect = (float(known_incorrect) / incorrect_labels.shape[0]) * 100
		unknown_incorrect = (float(unknown_incorrect) / incorrect_labels.shape[0]) * 100

		# Store the results for this openness
		known.append(known_incorrect)
		unknown.append(unknown_incorrect)

	# Plot the bar chart
	fig, ax = plt.subplots()
	ax.set_xlabel('Known / Unknown (%)')
	ax.set_ylabel('Incorrect Proportion (%)')
	ax.set_ylim((0, 100))

	x = np.arange(len(x_labels))
	width = 0.35  # the width of the bars
	rects1 = ax.bar(x - width/2, known, width, label='Known')
	rects2 = ax.bar(x + width/2, unknown, width, label='Unknown')

	ax.set_xticks(x)
	ax.set_xticklabels(x_labels)

	ax.legend()

	fig.tight_layout()
	# plt.show()
	plt.savefig("error-proportion.pdf")

	# Plot the other bar chart
	error_classes = labels_accum_incorrect / labels_accum_images * 100
	fig, ax = plt.subplots()
	ax.set_xlabel('ID')
	ax.set_ylabel('Testing Error (%)')
	ax.set_ylim((0, 40))
	ax.set_xlim((0,num_classes))
	plt.grid()
	ax.bar(np.arange(1,num_classes+1), error_classes)
	fig.tight_layout()
	# plt.show()
	plt.savefig("error-classes.pdf")

def plotTrainingAccuracy():
	# base_dir = "/home/will/work/CEiA/results/RTL/83-17"
	# base_dir = "/home/will/work/CEiA/results/SoftmaxTL/90-10"
	# base_dir = "/mnt/storage/home/ca0513/CEiA/results/RTL/50-50"
	# base_dir = "/home/will/work/CEiA/results/SoftmaxTL/50-50"
	# base_dir = "/mnt/storage/home/ca0513/CEiA/results/SoftmaxTL/75-25"
	# base_dir = "/home/ca0513/CEiA/results/SoftmaxRTL/75-25"
	base_dir = "D:\\Work\\results\\CEiA\\SRTL\\05"

	fold_folders = DataUtils.allFoldersAtDir(base_dir)
	max_folds = len(fold_folders)

	print(f"Results for: {base_dir}")

	# TEMP: remove the fold currently being computed
	# fold_folders.remove("/home/will/work/1-RA/src/Identifier/Supervised-Triplet-Network/results/fold_6")
	# max_folds -= 1

	# List of previous sizes
	acc_prev = []
	loss_prev = []

	for k in range(max_folds):
		accuracies_fp = os.path.join(fold_folders[k], "open_cows_triplet_cnn_accuracies_log_x1.npz")
		losses_fp = os.path.join(fold_folders[k], "open_cows_triplet_cnn_train_log_x1.npz")

		accuracies = np.load(accuracies_fp)
		losses = np.load(losses_fp)

		print(accuracies['accuracies_all'], accuracies['accuracies_known'])

		assert accuracies['accuracies_all'].shape == accuracies['accuracies_known'].shape 
		assert accuracies['accuracies_known'].shape == accuracies['accuracies_novel'].shape
		assert accuracies['accuracies_novel'].shape == accuracies['steps'].shape

		assert losses['losses_mean'].shape == losses['steps'].shape

		start_acc = int(np.sum(np.array(acc_prev)))
		end_acc = accuracies['steps'].shape[0]

		start_loss = int(np.sum(np.array(loss_prev)))
		end_loss = losses['steps'].shape[0]

		acc_prev.append(end_acc - start_acc)
		loss_prev.append(end_loss - start_loss)

		accuracy_all = accuracies['accuracies_all'][start_acc:end_acc]
		accuracies_known = accuracies['accuracies_known'][start_acc:end_acc]
		accuracies_novel = accuracies['accuracies_novel'][start_acc:end_acc]
		accuracy_steps = accuracies['steps'][start_acc:end_acc]

		loss_sum = losses['losses_mean'][start_loss:end_loss]
		loss_steps = losses['steps'][start_loss:end_loss]

		plotAccuracyLoss(	k, 
							accuracy_all, 
							accuracies_known, 
							accuracies_novel, 
							accuracy_steps,
							loss_sum,
							loss_steps			)

if __name__ == '__main__':
	# plotAccuracyVsUnknownRatio()
	# plotAccuracyVsFolds()
	# plotErrorOrigin()
	plotTrainingAccuracy()
