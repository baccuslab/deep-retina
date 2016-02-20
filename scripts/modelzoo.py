import numpy as np
import os
import csv
import json

new_csv_name = raw_input("Please type in the new csv name and press 'Enter' (e.g. models.csv): ")
f = raw_input("Please type in the path you want to start from and press 'Enter': ")

if f[0] == '~':
    f = os.path.expanduser(f)

print('Starting to walk down directories')
# e.g. first run from os.path.expanduser('~/Dropbox/deep-retina/saved')
walker = os.walk(f, topdown=True)
architecture_name = 'architecture.json'
performance_name = 'performance.csv'
metadata_name = 'metadata.json'
readme_name = 'README.md'
experiment_name = 'experiment.json'

models_parsed = 0
all_models = []
for dirs, subdirs, files in walker:
    # select only model folders
    if architecture_name in files:
        # open model architecture file
        with open(os.path.join(dirs, architecture_name), 'r') as f:
            arch = json.load(f)

        subparts = dirs.split('/')
        model_id = subparts[-1]

        # check if trained on deepretina v0.2
        if metadata_name in files:
            with open(os.path.join(dirs, metadata_name), 'r') as f:
                meta = json.load(f)

            date = meta['date']
            deepretina_ver = meta['deep-retina']
            machine = meta['machine']
            user = meta['user']

            with open(os.path.join(dirs, experiment_name), 'r') as f:
                expt = json.load(f)

            experiment = expt['date']
            stimulus = expt['filename']
            history = expt['history']
            cells = expt['cells']


            # Parse train.csv and test.csv
            train_path = os.path.join(dirs, 'train.csv')
            test_path = os.path.join(dirs, 'test.csv')
            with open(train_path, 'r') as h:
                train_stats = csv.reader(h)
                train_rows = []
                for row in train_stats:
                    train_rows.append(row)

            with open(test_path, 'r') as h:
                test_stats = csv.reader(h)
                test_rows = []
                for row in test_stats:
                    test_rows.append(row)


            train_table = np.array(train_rows)
            test_table = np.array(test_rows)

            # handle models that never trained
            try:
                num_epochs = np.array(train_rows)[-1, 0].astype('int')
            except:
                num_epochs = 0

            if num_epochs > 0:
                # avoid header
                train_numbers = train_table[1:, :].astype('float')
                test_numbers = test_table[1:, :].astype('float')

                max_train_cc = np.nanmax(train_numbers[:,2])
                mean_train_cc = np.nanmean(train_numbers[:,2])
                max_test_cc = np.nanmax(test_numbers[:,2])
                mean_test_cc = np.nanmean(test_numbers[:,2])
            else:
                max_train_cc = 0
                mean_train_cc = 0
                max_test_cc = 0
                mean_test_cc = 0
            
        # otherwise we're stuck with dark ages and have
        # to manually parse README
        else:
            deepretina_ver = '<0.2'
            history = 40
            # Parse README.md
            readme_path = os.path.join(dirs, readme_name)
            with open(readme_path, 'r') as f:
                description = f.readlines()
            path_components = readme_path.split('/')
            for idd, d in enumerate(description):
                if d.find('Cell') >= 0:
                    cells = d[:-1]
                elif d.find('Stimulus') >= 0:
                    experiment = description[idd+1][:-1]
                    stimulus = description[idd+2][:-1]

            date = description[1][3:-1]
            machine = path_components[-3]
            user = path_components[-2]

            # Parse performance.csv
            performance_path = os.path.join(dirs, 'performance.csv')
            with open(performance_path, 'r') as h:
                stats = csv.reader(h)
            all_rows = []
            for row in stats:
                all_rows.append(row)
            num_epochs = len(all_rows) - 1 # -1 for the header

            # skip models for which we have no performance data
            if num_epochs >= 1:
                table = np.array(all_rows)
                just_numbers = table[1:, :].astype('float')

            max_train_cc = np.nanmax(just_numbers, axis=0)[2]
            mean_train_cc = np.nanmean(just_numbers, axis=0)[2]
            max_test_cc = np.nanmax(just_numbers, axis=0)[3]
            mean_test_cc = np.nanmean(just_numbers, axis=0)[3]

        # save layers of model
        layers = arch['layers']
        loss = arch['loss']

        layer_names = [l['name'] for l in layers]
        if 'LSTM' in layer_names:
            model_type = 'fixedlstm'
        elif 'TimeDistributedConvolution2D' in layer_names:
            model_type = 'lstm'
        else:
            model_type = 'convnet' 

        if 'Dropout' in layer_names:
            dropout = [l['p'] for l in layers if 'Dropout' in l['name']][0]
        else:
            dropout = 0

        l2 = layers[0]['W_regularizer']['l2']
        
        

        model = {
            'model_id': model_id,
            'type': model_type, # from architecture.json
            'date': date,
            'machine': machine,
            'user': user,
            'stimulus': stimulus,
            'experiment': experiment,
            'cells': cells,
            'epochs': num_epochs,
            'loss': loss,
            'dropout': dropout,
            'l2': l2,
            'nlayers': len(layers),
            'nconvlayers': len([l for l in layers if 'Convolution2D' in l['name']]),
            'naffinelayers': len([l for l in layers if 'Dense' in l['name']]),
            'npoollayers': len([l for l in layers if 'MaxPooling2D' in l['name']]),
            'nfilters': [l['nb_filter'] for l in layers if 'Convolution2D' in l['name']],
            'filtershapes': [l['nb_row'] for l in layers if 'Convolution2D' in l['name']],
            'layers': [l['name'] for l in layers],
            'history': history,
            'deepretina version': deepretina_ver,
            'max_train_cc': max_train_cc,
            'mean_train_cc': mean_train_cc,
            'max_test_cc': max_test_cc,
            'mean_test_cc': mean_test_cc
        }
        all_models.append(model)
        models_parsed += 1
        print('Saved model %i' %(models_parsed))
    
header = {}
for k in sorted(all_models[0].keys()):
    header[k] = k

all_models.insert(0, header)

g = open(new_csv_name,'w')
w = csv.DictWriter(g, sorted(all_models[0].keys()))
w.writerows(all_models)
g.close()
