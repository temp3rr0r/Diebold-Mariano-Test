# MIT License
# Copyright (c) 2017 John Tsang
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Sample script modified by Konstantinos Theodorakos on 14 September 2021

from dm_test import dm_test
import random
import numpy as np

# See: https://pkg.robjhyndman.com/forecast/reference/dm.test.html#author
print("-- Diebold-Mariano (DM): statistical significance test for predictive accuracy comparisons (with p-value as the significance level) ---")
print("Rationale: Given two forecasts of horizon h, and the expected values:")
print("\tIf DM close to zero -> The two predictions are similar (null hypothesis is TRUE, p-value should be high)")
print("\tif DM > 0 -> second more accurate,")
print("\tif DM < 0 -> first more accurate.")
print("\tif DM < 0 -> first more accurate.")
# See: https://en.wikipedia.org/wiki/Misuse_of_p-values
print("\tA low p-value means...")
print("\teither null hypothesis TRUE + a highly improbable event happened")
print("\tor null hypothesis is FALSE")


def print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst):
    elements_to_show = 10
    print(f"Actual time-series: \t{np.round(np.array(actual_lst[0:elements_to_show], dtype=np.float32), 1)}, ...")
    print(f"1st prediction: \t\t{np.round(np.array(pred1_lst[0:elements_to_show]), 2)}, ...")
    print(f"2nd prediction: \t\t{np.round(np.array(pred2_lst[0:elements_to_show]), 2)}, ...")

    dms = []
    ps = []
    rt = dm_test(actual_lst, pred1_lst, pred2_lst, h=horizon, crit="MAD")
    print("MAD", rt)
    dms.append(rt.DM)
    ps.append(rt.p_value)
    rt = dm_test(actual_lst, pred1_lst, pred2_lst, h=horizon, crit="MSE")
    print("MSE", rt)
    dms.append(rt.DM)
    ps.append(rt.p_value)
    rt = dm_test(actual_lst, pred1_lst, pred2_lst, h=horizon, crit="poly", power=4)
    print("poly 4", rt)
    dms.append(rt.DM)
    ps.append(rt.p_value)

    decision = ""
    if np.abs(np.mean(dms)) < 4:
        decision = "Similar (null hypothesis TRUE)"
        if np.mean(ps) >= 0.5:
            decision += ", p-value HIGH (we are confident)"
        else:  # See: https://en.wikipedia.org/wiki/P-value, https://en.wikipedia.org/wiki/Misuse_of_p-values
            decision += ", p-value LOW (we are NOT confident: either null hypothesis FALSE or null hypothesis TRUE + highly improbable event happened)"
    else:
        if np.mean(dms) <= 0:
            decision = "First better"
        else:
            decision = "Second better"
    print(f"== Decision: {decision} (mean DM: {np.round(np.mean(dms), 2)}, mean p: {np.mean(ps)})")

# Parameters
ts_count = 100
horizon = 24

random.seed(123)
actual_lst = range(1, ts_count)
pred1_lst = range(1, ts_count)
pred2_lst = range(1, ts_count)

# actual_lst = random.sample(actual_lst, 100)
# pred1_lst = random.sample(pred1_lst, 100)
# pred2_lst = random.sample(pred2_lst, 100)

print(f"\nExamples (forecast horizon: {horizon}):")

print("\n--- Example data: Similar predictions (very similar):")
pred1_lst = []
mu, sigma = 0.1, 0.01  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred1_lst.append(actual_lst[i] + s[i])
pred2_lst = []
dms = []
ps = []
mu, sigma = 0.1, 0.01  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(actual_lst[i] + s[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst)

print("\n--- Example data: Similar predictions (not so similar):")
pred1_lst = []
mu, sigma = 0.5, 3  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred1_lst.append(actual_lst[i] + s[i])
pred2_lst = []
dms = []
ps = []
mu, sigma = -0.5, 3  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(actual_lst[i] + s[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst)

print("\n--- Example data: 1st prediction is (slightly) better:")
pred2_lst = []
mu, sigma = 1, 1.0  # 0, 0.1  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(pred1_lst[i] + s[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst)

print("\n--- Example data: 1st prediction is (much) better:")
pred2_lst = []
mu, sigma = 10, 3  # 0, 0.1  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(pred1_lst[i] + s[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst)

print("\n--- Example data: 2nd prediction is (slightly) better:")
pred2_lst = []
mu, sigma = 10, 3  # 0, 0.1  # mean and standard deviation
s2 = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(pred1_lst[i] + s2[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred2_lst, pred1_lst)
