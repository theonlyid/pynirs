% Import the library or a module (here we import the hb_conv module)
lib = py.importlib.import_module("pynirs.hb_conv")


% import the data and convert to Numpy array
data = readmatrix('../../../src/pynirs/data/test_data.csv');
data_np = lib.np.array(data');

% Pass numpy object to library and get results
res = lib.HbConv(data_np);
toi_observed = res.observed{'toi'};
toi_clean_abs = res.cleaned{'toi_abs'};
toi_clean_ods = res.cleaned{'toi_abs'};


% convert to matlab double and plot
toi_observed = double(toi_observed);
toi_clean_abs = double(toi_clean_abs);
figure(); hold on;
plot(toi_observed)
plot(toi_clean_abs)