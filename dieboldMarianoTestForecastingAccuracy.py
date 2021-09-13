# MIT License
# Copyright (c) 2017 John Tsang
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# Sample script

from dm_test import dm_test
import random
import numpy as np

print("-- Diebold-Mariano test for predictive accuracy ---")
print("Rationale: If DM close to zero -> The two predictions are similar (p signifies how confident)")
print("\tif DM > 0 -> second more accurate,")
print("\tif DM < 0 -> first more accurate.")
# See: https://pkg.robjhyndman.com/forecast/reference/dm.test.html#author

def print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst):
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
    if np.abs(np.mean(dms)) < 5:
        decision = "Similar"
    else:
        if np.mean(dms) <= 0:
            decision = "First better"
        else:
            decision = "Second better"
    print(f"== Decision: {decision} (mean DM: {np.round(np.mean(dms), 2)} mean p: {np.mean(ps)})")

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

print("Similar")
pred1_lst = []
mu, sigma = 0, 0.01  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred1_lst.append(actual_lst[i] + s[i])
pred2_lst = []
dms = []
ps = []
mu, sigma = 0, 0.01  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(actual_lst[i] + s[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst)

print("\n1st (slightly) better")
pred2_lst = []
mu, sigma = 0, 0.01  # 0, 0.1  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(pred1_lst[i] + s[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst)

print("\n1st (much) better")
pred2_lst = []
mu, sigma = 10, 3  # 0, 0.1  # mean and standard deviation
s = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(pred1_lst[i] + s[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred1_lst, pred2_lst)

print("\n2nd (much) better")
pred2_lst = []
mu, sigma = 10, 3  # 0, 0.1  # mean and standard deviation
s2 = np.random.normal(mu, sigma, len(actual_lst))
for i in range(len(actual_lst)):
    pred2_lst.append(pred1_lst[i] + s2[i])
print_diebold_mariano_predictive_accuracy(actual_lst, pred2_lst, pred1_lst)
