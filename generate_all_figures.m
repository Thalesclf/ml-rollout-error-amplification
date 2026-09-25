%% generate_all_figures.m
% Regenerate the paper figures from the saved campaign CSV files.
%
% Expected directory structure:
%   results/
%     seed_0/*.csv
%     seed_1/*.csv
%     ...
%
% Usage:
%   1. Put this script next to the "results" directory, OR change resultsDir.
%   2. Run this script in MATLAB.
%   3. Figures are written as vector PDFs to figures/.
%
% No PyTorch checkpoints are needed for these figures.
% Requires MATLAB with Statistics and Machine Learning Toolbox for
% corr(...,'Type','Spearman') and prctile.

clear; close all; clc;

%% ---------------- USER SETTINGS -----------------------------------------
resultsDir = fullfile(pwd, "results");
figDir     = fullfile(pwd, "figures");

if ~isfolder(resultsDir)
    error("Could not find results directory: %s", resultsDir);
end
if ~isfolder(figDir)
    mkdir(figDir);
end

% Paper plotting defaults.
set(groot, ...
    'defaultAxesFontName','Helvetica', ...
    'defaultAxesTickLabelInterpreter','latex', ...
    'defaultLegendInterpreter','latex', ...
    'defaultTextInterpreter','latex');


%% ---------------- LOAD AND POOL ALL SEEDS -------------------------------
D = readCampaign(resultsDir, "diagnostic_campaign.csv");
P = readCampaign(resultsDir, "perturbation_campaign.csv");
E = readCampaign(resultsDir, "epsilon_campaign.csv");
S = readCampaign(resultsDir, "spectral_modes_campaign.csv");
F = readCampaign(resultsDir, "defect_terms_campaign.csv");

fprintf("Loaded:\n");
fprintf("  diagnostic_campaign: %d rows\n", height(D));
fprintf("  perturbation_campaign: %d rows\n", height(P));
fprintf("  epsilon_campaign: %d rows\n", height(E));
fprintf("  spectral_modes_campaign: %d rows\n", height(S));
fprintf("  defect_terms_campaign: %d rows\n\n", height(F));

horizons = unique(D.horizon(:))';
horizons = sort(horizons);

%% FIG. 2b -- local and finite-time diagnostics across all horizons

horizons = sort(unique(D.horizon));

rho_Gmax   = nan(size(horizons));
rho_rhoJ   = nan(size(horizons));
rho_sigmaJ = nan(size(horizons));
rho_froJ   = nan(size(horizons));

for ii = 1:numel(horizons)

    T = horizons(ii);
    q = D.horizon == T;

    % Spearman correlation between each diagnostic and rollout error
    rho_Gmax(ii) = spearman( ...
        D.Gmax(q), ...
        D.rollout_error(q));

    rho_rhoJ(ii) = spearman( ...
        D.spectral_radius_J(q), ...
        D.rollout_error(q));

    rho_sigmaJ(ii) = spearman( ...
        D.sigma1_J(q), ...
        D.rollout_error(q));

    rho_froJ(ii) = spearman( ...
        D.fro_J(q), ...
        D.rollout_error(q));
end

%% Plot

f = paperFigure(396, 220);

hold on;

h1 = semilogx(horizons, rho_Gmax, '-o');
h1.MarkerFaceColor = h1.Color;

h2 = semilogx(horizons, rho_rhoJ, '-s');
h2.MarkerFaceColor = h2.Color;

h3 = semilogx(horizons, rho_sigmaJ, '-^');
h3.MarkerFaceColor = h3.Color;

h4 = semilogx(horizons, rho_froJ, '-d');
h4.MarkerFaceColor = h4.Color;

yline(0, '--');

grid on;
box on;

xlabel('Forecast horizon ($T$)');
ylabel('Spearman rank correlation ($\rho_s$)');

xticks(horizons);
xticklabels(string(horizons));

legend( ...
    {'$G_{\max}$', ...
     '$\rho(J_t)$', ...
     '$\sigma_1(J_t)$', ...
     '$\|J_t\|_F$'}, ...
    'Location', 'northwest');

set(gca, 'XScale', 'log');

exportPaper(f, figDir, ...
    "fig2b_local_vs_finite_time_metrics.pdf");

%% FIG. 3 -- Distribution of finite-time amplification across horizons

horizons = sort(unique(D.horizon));

% Create figure at final LaTeX size
f = paperFigure(396, 220);

% Boxplots
boxplot(D.Gmax, D.horizon, ...
    'Positions', 1:numel(horizons), ...
    'Labels', string(horizons), ...
    'Symbol', '');   % Hide individual outlier markers

% Logarithmic vertical axis
set(gca, 'YScale', 'log');

% Axis labels
xlabel('Forecast horizon ($T$)');
ylabel('$G_{\max}$');

% Grid and axes
grid on;
box on;

ax = gca;
ax.FontSize = 9;
ax.LineWidth = 0.75;

% Keep major grid subtle
ax.GridAlpha = 0.15;

% Disable visually dense minor grid
ax.XMinorGrid = 'off';
ax.YMinorGrid = 'off';

% Make tick marks point outward
ax.TickDir = 'out';

% Ensure discrete horizon labels
xticks(1:numel(horizons));
xticklabels(string(horizons));

set(gca,'TickLabelInterpreter','latex');

% Export as vector PDF
exportPaper(f, figDir, ...
    "fig3_horizon_dependence.pdf");

%% FIG. 4 -- perturbation comparison
% Pool all perturbation experiments over seeds.

labels = {'TG', '2nd singular', 'Random mean', 'PGD'};

Y = [ ...
    P.G_TG, ...
    P.G_v2, ...
    P.G_random_mean, ...
    P.G_PGD ...
];

f = paperFigure(396, 220);

% Standard boxplots for the four perturbation directions
boxplot(Y, ...
    'Labels', labels, ...
    'Symbol', '');

grid on;
box on;

xlabel('Perturbation');
ylabel('Nonlinear gain ($G_{\mathrm{NL}}$)');

ax = gca;
ax.FontSize = 9;
ax.LineWidth = 0.75;
ax.GridAlpha = 0.15;
ax.XMinorGrid = 'off';
ax.YMinorGrid = 'off';
ax.TickDir = 'out';
set(gca,'TickLabelInterpreter','latex');

exportPaper(f, figDir, ...
    "fig4_perturbation_comparison.pdf");

%% FIG. 5a -- TG/adversarial misalignment versus epsilon

panelWidth  = 0.49 * 396;
panelHeight = 130;

markerSize = 3.2;
lineWidth  = 0.9;
fontSize   = 7.5;

epsVals = sort(unique(E.epsilon));

misalignment = 1 - E.alignment;

[med,p10,p90] = groupedQuantiles( ...
    E.epsilon, misalignment, epsVals);

tiny = 1e-16;
med_plot = max(med,tiny);
p10_plot = max(p10,tiny);
p90_plot = max(p90,tiny);

f = paperFigure(panelWidth,panelHeight);
hold on;

% 10--90% interval
fill([epsVals(:); flipud(epsVals(:))], ...
     [p10_plot(:); flipud(p90_plot(:))], ...
     [0.5 0.5 0.5], ...
     'EdgeColor','none', ...
     'FaceAlpha',0.20);

% Median
h1 = plot(epsVals,med_plot,'-o', ...
    'LineWidth',lineWidth, ...
    'MarkerSize',markerSize);

h1.MarkerFaceColor = h1.Color;

set(gca,'XScale','log','YScale','log');

grid on;
box on;

ax = gca;
ax.FontSize = fontSize;
ax.LineWidth = lineWidth;

xlabel('$\epsilon$');
ylabel('$1-|v_{\mathrm{TG}}^\top v_{\mathrm{adv}}|$');

xlim([5e-7 2e-1]);

lgd = legend({'10--90\%','median'}, ...
    'Location','northwest');
lgd.FontSize = fontSize;

exportPaper(f,figDir, ...
    "fig5a_alignment_vs_epsilon_aggregate.pdf");

%% FIG. 5b -- normalized nonlinear gain versus epsilon

ratioTG  = E.G_NL_TG  ./ E.sigma1_sq;
ratioPGD = E.G_NL_PGD ./ E.sigma1_sq;

[mTG,lTG,uTG] = groupedQuantiles( ...
    E.epsilon,ratioTG,epsVals);

[mPG,lPG,uPG] = groupedQuantiles( ...
    E.epsilon,ratioPGD,epsVals);

f = paperFigure(panelWidth,panelHeight);
hold on;

% TG 10--90% band
fillBandLogX(epsVals,lTG,uTG,0.14);

% TG median
h1 = semilogx(epsVals,mTG,'-o', ...
    'LineWidth',lineWidth, ...
    'MarkerSize',markerSize);
h1.MarkerFaceColor = h1.Color;

% PGD median
h2 = semilogx(epsVals,mPG,'-o', ...
    'LineWidth',lineWidth, ...
    'MarkerSize',markerSize);
h2.MarkerFaceColor = h2.Color;

% Infinitesimal prediction
yline(1,'--', ...
    'LineWidth',lineWidth);

grid on;
box on;

ax = gca;
ax.FontSize = fontSize;
ax.LineWidth = lineWidth;

xlabel('$\epsilon$');
ylabel('$G_{\mathrm{NL}}/\sigma_1^2(\Phi)$');

xlim([5e-7 2e-1]);

lgd = legend( ...
    {'TG 10--90\%','TG','PGD','Linear'}, ...
    'Location','northwest');
lgd.FontSize = fontSize;

exportPaper(f,figDir, ...
    "fig5b_nonlinear_gain_vs_epsilon_aggregate.pdf");

%% FIG. 6a -- linearized prediction versus actual rollout error

horizons = sort(unique(D.horizon));

f = paperFigure(396,220);
hold on;

% One scatter series per horizon
h = gobjects(numel(horizons),1);

for ii = 1:numel(horizons)

    T = horizons(ii);

    q = D.horizon == T & ...
        isfinite(D.rollout_error) & ...
        isfinite(D.linear_predicted_error) & ...
        D.rollout_error > 0 & ...
        D.linear_predicted_error > 0;

    h(ii) = scatter( ...
        D.rollout_error(q), ...
        D.linear_predicted_error(q), ...
        8, ...
        'filled', ...
        'MarkerFaceAlpha',0.35, ...
        'DisplayName',sprintf('$T=%g$',T));
end

% Identity line y = x
allActual = D.rollout_error( ...
    isfinite(D.rollout_error) & D.rollout_error > 0);

allPred = D.linear_predicted_error( ...
    isfinite(D.linear_predicted_error) & D.linear_predicted_error > 0);

lo = min([allActual; allPred]);
hi = max([allActual; allPred]);

href = loglog([lo hi],[lo hi],'--', ...
    'LineWidth',1.0, ...
    'DisplayName','$y=x$');

% Logarithmic axes
set(gca,'XScale','log','YScale','log');

grid on;
box on;

ax = gca;
ax.FontSize = 9;
ax.LineWidth = 0.75;
ax.GridAlpha = 0.15;
ax.MinorGridAlpha = 0.05;
ax.TickDir = 'out';

xlabel('Actual rollout error ($\|\delta_T\|$)');
ylabel('Linearized prediction ($\|\delta_T^{\mathrm{lin}}\|$)');

% Keep equal numerical limits so the identity line has a clear meaning
xlim([lo hi]);
ylim([lo hi]);

legend(h,'Location','northwest');

exportPaper(f,figDir, ...
    "fig6a_linear_error_prediction.pdf");

%% FIG. 6b -- error-injection diagnostics across forecast horizons

horizons = sort(unique(D.horizon));

rho_Gmax       = nan(size(horizons));
rho_defect     = nan(size(horizons));
rho_propagated = nan(size(horizons));
rho_vector     = nan(size(horizons));
rho_bound      = nan(size(horizons));

for ii = 1:numel(horizons)

    T = horizons(ii);
    q = D.horizon == T;

    y = D.rollout_error(q);

    % Worst-case finite-time amplification
    rho_Gmax(ii) = spearman( ...
        D.Gmax(q), y);

    % Sum of local defect magnitudes
    rho_defect(ii) = spearman( ...
        D.defect_l2_sum(q), y);

    % Sum of individually propagated defect magnitudes
    rho_propagated(ii) = spearman( ...
        D.propagated_defect_norm_sum(q), y);

    % Norm of the vector sum of propagated defects
    rho_vector(ii) = spearman( ...
        D.linear_predicted_error(q), y);

    % Worst-case defect-amplification bound
    rho_bound(ii) = spearman( ...
        D.defect_amplification_bound(q), y);
end

%% FIG. 6b -- error-injection diagnostics across forecast horizons

f = paperFigure(396,220);
hold on;

h1 = plot(horizons,rho_Gmax,'-o');
h1.MarkerFaceColor = h1.Color;

h2 = plot(horizons,rho_defect,'-o');
h2.MarkerFaceColor = h2.Color;

h3 = plot(horizons,rho_propagated,'-o');
h3.MarkerFaceColor = h3.Color;

h4 = plot(horizons,rho_vector,'-o');
h4.MarkerFaceColor = h4.Color;

h5 = plot(horizons,rho_bound,'-o');
h5.MarkerFaceColor = h5.Color;

grid on;
box on;

xlabel('Forecast horizon ($T$)');
ylabel('Spearman rank correlation ($\rho_s$)');

xticks(horizons);
xlim([0 max(horizons)]);

ylim([0 1.02]);

legend({ ...
    '$G_{\max}$', ...
    '$\sum_k \|e_k\|$', ...
    '$\sum_k \|\Phi_k e_k\|$', ...
    '$\|\sum_k \Phi_k e_k\|$', ...
    '$B_T$'}, ...
    'Location','northwest');

set(gca,'XScale','log');

exportPaper(f,figDir, ...
    "fig6b_error_injection_diagnostics.pdf");

%% FIG. 6c -- linearization accuracy versus forecast horizon
% Distribution of ||delta_T^lin|| / ||delta_T|| at each horizon.

horizons = sort(unique(D.horizon));

% Remove invalid values
q = isfinite(D.linear_to_actual_ratio) & ...
    isfinite(D.horizon) & ...
    D.linear_to_actual_ratio > 0;

ratio = D.linear_to_actual_ratio(q);
Tdata = D.horizon(q);

f = paperFigure(396,220);
hold on;

% Boxplot at each discrete forecast horizon.
% Using positions 1,...,N gives equal horizontal spacing between horizons.
boxplot(ratio, Tdata, ...
    'Positions', 1:numel(horizons), ...
    'Labels', string(horizons), ...
    'Symbol', '');

% Perfect-agreement reference
yline(1, '--');

grid on;
box on;

xlabel('Forecast horizon ($T$)');
ylabel('$R_{\mathrm{lin}}$');
ylim([0 8]);

% Explicit categorical positions
xticks(1:numel(horizons));
xticklabels(string(horizons));

set(gca,'TickLabelInterpreter','latex');

exportPaper(f,figDir, ...
    "fig6c_linearization_accuracy_vs_horizon.pdf");

%% FIG. 7a -- defect alignment versus gain efficiency
% Density representation of all defect-level observations.

q = isfinite(F.tg_alignment) & ...
    isfinite(F.gain_efficiency);

a   = F.tg_alignment(q);
eta = F.gain_efficiency(q);

f = paperFigure(396,220);
hold on;

nBins = 100;

xedges = linspace(0,1,nBins+1);
yedges = linspace(0,1,nBins+1);

N = histcounts2(a,eta,xedges,yedges);

xc = (xedges(1:end-1) + xedges(2:end))/2;
yc = (yedges(1:end-1) + yedges(2:end))/2;

%imagesc(xc,yc,log10(N' + 1));
pcolor(xc,yc,log10(N' + 1));
shading interp;
colormap(jet);
set(gca,'Layer','top');

set(gca,'YDir','normal');


hLower = plot([0 1],[0 1],'--');


hUpper = yline(1,'--');


xlim([0 1]);
ylim([0 1.02]);

grid on;
box on;

xlabel('Defect--TG alignment ($a_k$)');
ylabel('Gain efficiency ($\eta_k$)');

cb = colorbar;
cb.Label.String = '$\log_{10}(N+1)$';
cb.Label.Interpreter = 'latex';
cb.TickLabelInterpreter = 'latex';

legend([hLower hUpper], ...
    {'$\eta_k=a_k$', '$\eta_k=1$'}, ...
    'Location','southeast');

exportPaper(f,figDir, ...
    "fig7a_defect_alignment_vs_gain_efficiency.pdf");

%% FIG. 7b -- error-aligned diagnostics across horizons

horizons = sort(unique(D.horizon));

rho_defect     = nan(size(horizons));
rho_propagated = nan(size(horizons));
rho_vector     = nan(size(horizons));
rho_effgain    = nan(size(horizons));
rho_alignment  = nan(size(horizons));

for ii = 1:numel(horizons)

    T = horizons(ii);
    q = D.horizon == T;

    y = D.rollout_error(q);

    rho_defect(ii) = spearman( ...
        D.defect_l2_sum(q), y);

    rho_propagated(ii) = spearman( ...
        D.propagated_defect_norm_sum(q), y);

    rho_vector(ii) = spearman( ...
        D.linear_predicted_error(q), y);

    rho_effgain(ii) = spearman( ...
        D.effective_defect_gain(q), y);

    rho_alignment(ii) = spearman( ...
        D.tg_alignment_weighted(q), y);
end

%% Plot

f = paperFigure(396,220);
hold on;

h1 = plot(horizons,rho_defect,'-o');
h1.MarkerFaceColor = h1.Color;

h2 = plot(horizons,rho_propagated,'-o');
h2.MarkerFaceColor = h2.Color;

h3 = plot(horizons,rho_vector,'-o');
h3.MarkerFaceColor = h3.Color;

h4 = plot(horizons,rho_effgain,'-o');
h4.MarkerFaceColor = h4.Color;

h5 = plot(horizons,rho_alignment,'-o');
h5.MarkerFaceColor = h5.Color;

grid on;
box on;

xlabel('Forecast horizon ($T$)');
ylabel('Spearman rank correlation ($\rho_s$)');

set(gca,'XScale','log');
xticks(horizons);
xticklabels(string(horizons));

xlim([1 160]);
ylim([-0.1 1.02]);

legend({ ...
    '$\sum_k\|e_k\|$', ...
    '$\sum_k\|\Phi_k e_k\|$', ...
    '$\|\sum_k\Phi_k e_k\|$', ...
    'Effective defect gain', ...
    'Weighted TG alignment'}, ...
    'Location','southwest');

exportPaper(f,figDir, ...
    "fig7b_error_aligned_diagnostics.pdf");

%% FIG. 7c -- defect cancellation versus forecast horizon

horizons = sort(unique(D.horizon));

q = isfinite(D.horizon) & ...
    isfinite(D.defect_cancellation_ratio);

Tdata = D.horizon(q);
CT    = D.defect_cancellation_ratio(q);

f = paperFigure(396,220);
hold on;

% Distribution at each forecast horizon
boxplot(CT, Tdata, ...
    'Positions',1:numel(horizons), ...
    'Labels',string(horizons), ...
    'Symbol','');

grid on;
box on;

xlabel('Forecast horizon ($T$)');
ylabel('Cancellation ratio ($C_T$)');

xticks(1:numel(horizons));
xticklabels(string(horizons));

ylim([0 1.03]);

exportPaper(f,figDir, ...
    "fig7c_defect_cancellation_vs_horizon.pdf");

%% FIG. 8a -- cumulative realized singular amplification
% Use all non-degenerate modal decompositions. For each defect-tail pair,
% cumulative_realized_fraction already contains P_m.
q = isfinite(S.cumulative_realized_fraction);
SS = S(q,:);

modes = sort(unique(SS.mode));
medP = nan(size(modes));
p10P = nan(size(modes));
p90P = nan(size(modes));

for ii = 1:numel(modes)
    z = SS.cumulative_realized_fraction(SS.mode == modes(ii));
    z = z(isfinite(z));
    medP(ii) = median(z);
    p10P(ii) = prctile(z,10);
    p90P(ii) = prctile(z,90);
end

f = paperFigure(396,220);
hold on;
fillBand(modes,p10P,p90P,0.18);
plot(modes,medP,'-o','MarkerSize',3);
yline(0.9,'--');
grid on; box on;
xlim([min(modes),max(modes)]);
ylim([0,1.02]);
xlabel('Number of leading singular directions ($m$)');
ylabel('Cumulative realized amplification ($P_m$)');
legend({'10--90\%','Median','90\%'},'Location','southeast');
exportPaper(f, figDir, "fig8a_cumulative_realized_singular_amplification.pdf");

%% FIG. 8b -- realized spectral dimensionality at longest horizon
Tlong = max(F.horizon);
q = F.horizon == Tlong & F.modes_for_90pct > 0 & ...
    isfinite(F.realized_participation_modes);

f = paperFigure(396,220);
tl = tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

ax = nexttile(tl);
histogram(ax,F.modes_for_90pct(q),'BinMethod','integers');
grid(ax,'on'); box(ax,'on');
xlabel(ax,'Modes required for $P_m\geq0.9$');
ylabel(ax,'count');

ax = nexttile(tl);
histogram(ax,F.realized_participation_modes(q),25);
grid(ax,'on'); box(ax,'on');
xlabel(ax,'Realized participation number');
ylabel(ax,'count');

title(tl,sprintf('Spectral dimensionality of realized amplification, $T=%g$',Tlong));
exportPaper(f, figDir, "fig8b_realized_spectral_dimensionality.pdf");

fprintf("\nAll figures written to:\n  %s\n", figDir);

%% ======================== LOCAL FUNCTIONS ===============================

function T = readCampaign(resultsDir,fileName)
% Concatenate every seed_*/fileName table.
    listing = dir(fullfile(resultsDir,"seed_*",fileName));
    if isempty(listing)
        error("No files named %s found under %s/seed_*",fileName,resultsDir);
    end

    tables = cell(numel(listing),1);
    for i = 1:numel(listing)
        p = fullfile(listing(i).folder,listing(i).name);
        tables{i} = readtable(p,'VariableNamingRule','preserve');

        % If a per-seed CSV does not itself contain a seed column, recover it
        % from the parent directory name.
        if ~ismember("seed",string(tables{i}.Properties.VariableNames))
            tok = regexp(listing(i).folder,'seed_(\d+)','tokens','once');
            if ~isempty(tok)
                tables{i}.seed = repmat(str2double(tok{1}),height(tables{i}),1);
            end
        end
    end
    T = vertcat(tables{:});
end

function r = spearman(x,y)
% Pairwise-complete Spearman rank correlation.
    q = isfinite(x) & isfinite(y);
    x = x(q); y = y(q);
    if numel(x) < 2
        r = NaN;
    else
        r = corr(x,y,'Type','Spearman','Rows','complete');
    end
end

function [med,p10,p90] = groupedQuantiles(group,value,levels)
% Median and 10th/90th percentiles at specified group levels.
    med = nan(size(levels));
    p10 = nan(size(levels));
    p90 = nan(size(levels));

    for i = 1:numel(levels)
        z = value(group == levels(i));
        z = z(isfinite(z));
        if ~isempty(z)
            med(i) = median(z);
            p10(i) = prctile(z,10);
            p90(i) = prctile(z,90);
        end
    end
end

function fillBand(x,lo,hi,alphaValue)
% Linear-x uncertainty band.
    q = isfinite(x) & isfinite(lo) & isfinite(hi);
    x=x(q); lo=lo(q); hi=hi(q);
    h = fill([x(:);flipud(x(:))], ...
             [lo(:);flipud(hi(:))], ...
             [0.5 0.5 0.5], ...
             'EdgeColor','none','FaceAlpha',alphaValue);
    uistack(h,'bottom');
end

function fillBandLogX(x,lo,hi,alphaValue)
% Uncertainty band compatible with logarithmic x-axis.
    q = isfinite(x) & x>0 & isfinite(lo) & isfinite(hi);
    x=x(q); lo=lo(q); hi=hi(q);
    h = fill([x(:);flipud(x(:))], ...
             [lo(:);flipud(hi(:))], ...
             [0.5 0.5 0.5], ...
             'EdgeColor','none','FaceAlpha',alphaValue);
    set(gca,'XScale','log');
    uistack(h,'bottom');
end

function f = paperFigure(widthIn,heightIn)
% Create a consistently sized white-background publication figure.
    f = figure('Color','w','Units','points', ...
        'Position',[1 1 widthIn heightIn], ...
        'PaperPositionMode','auto');
end

function exportPaper(f,figDir,fileName)
% Export vector PDF suitable for LaTeX inclusion.
    drawnow;
    exportgraphics(f,fullfile(figDir,fileName), ...
        'ContentType','vector','BackgroundColor','white');
    close(f);
end
