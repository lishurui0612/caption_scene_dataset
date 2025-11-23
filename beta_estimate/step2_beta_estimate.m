function step2_beta_estimate(subject)
    %% Add GLMsingle in path
    root = '/public/home/lishr2022/Project/Cross-modal/beta_estimate';
    cd(root);
    cd("GLMsingle");
    setup
    cd(root)

    %% Initialization
    subjects = {'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8'};
    script_root = '/public/home/lishr2022/Project/Cross-modal/experiment/stimulus/scripts';
    response_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data';

    num_vertices = int32([149079, 151166;
                    135103, 135723;
                    155295, 151303;
                    141922, 142796;
                    141578, 138836;
                    146440, 149139;
                    145747, 144531;
                    129958, 128115]);
    response_length = 150;

    %% Read Stimulus index
    fid = fopen(fullfile(root, 'Stimulus_index.txt'));
    data = textscan(fid, '%d %s');
    fclose(fid);

    keys = data{2};
    values = num2cell(data{1});
    Stimulus_index = containers.Map(keys, values);

    %% GLM
    subject_id = str2num(subject(2));
    disp(subjects{subject_id});
    subject_script_root = fullfile(script_root, strcat('subject', num2str(subject_id)));
    subject_response_root = [fullfile(response_root, subjects{subject_id}) '/Stimulus/rescaled'];
    session_group = load(fullfile(root, strcat(subjects{subject_id},'.txt')));
    output_root = [fullfile(response_root, subjects{subject_id}) '/Stimulus'];
    beta_root = fullfile(response_root, subjects{subject_id}, 'Stimulus', 'beta');
    beta_zscore_root = fullfile(response_root, subjects{subject_id}, 'Stimulus', 'beta_zscore');

    if exist(subject_response_root, 'dir') ~= 7
        return;
    end

%    if exist(beta_root, 'dir') == 7 || exist(beta_zscore_root, 'dir') == 7
%        return;
%    end

    [total_run, total_group] = size(session_group);
    for group = 1:total_group
        response = {};
        design = {};
        group_run = {};
        % build design matrix
        for run = 1:200
            if session_group(run, group) == 0
                continue;
            end

            group_run{end+1} = run;

            run_script_root = fullfile(subject_script_root, ['subject' num2str(subject_id) '_run' num2str(run) '.txt']);
            design_matrix = zeros(300, Stimulus_index.Count);

            fid = fopen(run_script_root);
            header = fgetl(fid);
            data = textscan(fid, '%d %f %d %d %s %s', 'Delimiter', '\t');
            fclose(fid);

            [total_trial, i] = size(data{1});
            for trial = 1:total_trial
                if data{3}(trial) == 1
                    continue
                end
                Caption_onset_tr = int32(data{2}(trial));
                condition = data{6}(trial);
                condition = condition{1};
                design_matrix(Caption_onset_tr, Stimulus_index(condition)) = 1;

                Image_onset_tr = Caption_onset_tr + 6;
                condition = data{5}(trial);
                condition = condition{1};
                design_matrix(Image_onset_tr, Stimulus_index(condition)) = 1;
            end
            design{end+1} = design_matrix;

            % build response cell
            run_root = fullfile(subject_response_root, [subjects{subject_id} '_Run_' num2str(run, '%03d') '_scaled.mat']);
            run_response = load(run_root);
            run_response = squeeze(run_response.data);
            run_response = tseriesinterp(run_response, 2, 1, 2);
            response{end+1} = run_response;
        end
        disp('Successfully build response and design cells!');
        group_output = fullfile(output_root, strcat('Group', num2str(group)));

        if exist(group_output, 'dir')
            continue;
        end

        % build opt
        opt = struct;
        opt.sessionindicator = nonzeros(session_group(:, group));
        opt.wantglmdenoise = 1;
        opt.wantfracridge = 1;
        opt.wantfileoutputs = [0 0 0 1];
        opt.wantmemoryoutputs = [0 0 0 1];

        fprintf('Start GLMsingle for %s Group %d', subjects{subject_id}, group);
        [result, resultdesign] = GLMestimatesingletrial(design, response, 3, 1, group_output, opt);

        beta_root = fullfile(response_root, subjects{subject_id}, 'Stimulus', 'beta');
        if ~exist(beta_root, 'dir')
            mkdir(beta_root)
        end

        % 将每个trial的beta estimate单独保存
        modelmd = squeeze(result{4}.modelmd);
        HRFindex = result{4}.HRFindex;
        total_stim = squeeze(resultdesign.stimorder);
        for stim = 1:length(total_stim)
            run = 1;
            while sum(resultdesign.numtrialrun(1:run)) < stim
                run = run + 1;
            end
            run_beta_target = fullfile(beta_root, strcat(subjects{subject_id}, '_stim_', num2str(resultdesign.stimorder(stim), '%05d'), '_Run_', num2str(group_run{run}, '%03d'), '_beta.mat'));
            beta_data = modelmd(:, stim);
            if exist(run_beta_target)
                run_beta_target = fullfile(beta_root, strcat(subjects{subject_id}, '_stim_', num2str(resultdesign.stimorder(stim), '%05d'), '_Run_', num2str(group_run{run}, '%03d'), '_beta_2.mat'));
            end
            if mod(stim, 2) == 0
                pair_stim = resultdesign.stimorder(stim-1);
            else
                pair_stim = resultdesign.stimorder(stim+1);
            end
            save(run_beta_target, 'beta_data', 'HRFindex', 'pair_stim');
        end
        fprintf('Successfully save beta estimate for each trials.\n');
    end
end