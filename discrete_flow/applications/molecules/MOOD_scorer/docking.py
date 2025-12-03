#### Modified from https://github.com/hyanan16/SVDD-molecule/tree/main/MOOD_scorer
#### Reference for compiling Vina-GPU: https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1/tree/main/QuickVina2-GPU-2.1

#!/usr/bin/env python
import os
from shutil import rmtree
import subprocess
from openbabel import pybel
import multiprocessing as mp
import time
import numpy as np
def get_dockingvina(target, exhaustiveness=1, num_cpu=1, use_gpu=False, gpu_thread=8000, gpu_parallel=True, eval_batch_size=16, base_temp_dir='/tmp/yukang'):
    docking_config = dict()

    if target == 'jak2':
        box_center = (114.758,65.496,11.345)
        box_size= (19.033,17.929,20.283)
    elif target == 'braf':
        box_center = (84.194,6.949,-7.081)
        box_size = (22.032,19.211,14.106)
    elif target == 'fa7':
        box_center = (10.131, 41.879, 32.097)
        box_size = (20.673, 20.198, 21.362)
    elif target == 'parp1':
        box_center = (26.413, 11.282, 27.238)
        box_size = (18.521, 17.479, 19.995)
    elif target == '5ht1b':
        box_center = (-26.602, 5.277, 17.898)
        box_size = (22.5, 22.5, 22.5)

    docking_config['receptor_file']     = f'MOOD_scorer/receptors/{target}.pdbqt'
    docking_config['vina_program']      = 'MOOD_scorer/qvina02' if not use_gpu else 'QuickVina2-GPU-2.1/QuickVina2-GPU-2-1' #'/scratch/gpfs/yy1325/codes/DiscreteGuidance/Unlocking_Guidance/public_repo/discrete_guidance/applications/molecules/MOOD_scorer/QuickVina2-GPU-2-1'
    docking_config['box_parameter']     = (box_center, box_size)
    docking_config['exhaustiveness']    = exhaustiveness
    docking_config['num_sub_proc']      = 1 if use_gpu else 10
    docking_config['num_modes']         = 10
    docking_config['timeout_gen3d']     = 100
    docking_config['timeout_dock']      = 10000
    docking_config['use_gpu']          = use_gpu
    #### only applicable for Vina-cpu
    docking_config['num_cpu_dock']      = num_cpu
    #### [ADD] only applicable for Vina-GPU
    docking_config["opencl_binary_path"] = "./QuickVina2-GPU-2.1"
    docking_config["thread"] = gpu_thread # 8000
    docking_config["gpu_parallel"] = gpu_parallel
    docking_config["eval_batch_size"] = eval_batch_size
    docking_config['base_temp_dir'] = base_temp_dir
    return DockingVina(docking_config)


def make_docking_dir(base_temp_dir):
    assert os.path.exists(base_temp_dir), f"Base temp directory {base_temp_dir} does not exist."
    cur_time_stamp = int(time.time())
    for i in range(100):
        tmp_dir = f'{base_temp_dir}/{cur_time_stamp}_{i}'
        if not os.path.exists(tmp_dir):
            print(f'Docking tmp dir: {tmp_dir}')
            try:
                os.makedirs(tmp_dir)
                return tmp_dir
            except:
                continue
    raise ValueError('tmp/tmp0~99 are full. Please delete tmp dirs.')


class DockingVina(object):
    def __init__(self, docking_params):
        super(DockingVina, self).__init__()
        self.base_temp_dir = docking_params['base_temp_dir']
        self.temp_dir = make_docking_dir(self.base_temp_dir)
        # if not os.path.exists(self.temp_dir):
        #     os.makedirs(self.temp_dir)

        self.vina_program = docking_params['vina_program']
        self.receptor_file = docking_params['receptor_file']
        box_parameter = docking_params['box_parameter']
        (self.box_center, self.box_size) = box_parameter
        self.exhaustiveness = docking_params['exhaustiveness']
        self.num_sub_proc = docking_params['num_sub_proc']
        self.num_cpu_dock = docking_params['num_cpu_dock']
        self.num_modes = docking_params['num_modes']
        self.timeout_gen3d = docking_params['timeout_gen3d']
        self.timeout_dock = docking_params['timeout_dock']
        self.use_gpu = docking_params['use_gpu']
        self.opencl_binary_path = docking_params["opencl_binary_path"]
        self.thread = docking_params["thread"]
        self.gpu_parallel = docking_params["gpu_parallel"]
        self.eval_batch_size = docking_params["eval_batch_size"]
        print("using cpu docking" if not self.use_gpu else "using gpu docking")

    def gen_3d(self, smi, ligand_mol_file):
        run_line = 'obabel -:%s --gen3D -O %s' % (smi, ligand_mol_file)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_gen3d, universal_newlines=True)

    def docking(self, receptor_file, ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file):
        # assert not self.gpu_parallel, "gpu_parallel is not supported in this function."
        ms = list(pybel.readfile("mol", ligand_mol_file))
        m = ms[0]
        m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' % (self.box_size)
        if self.use_gpu:
            run_line += ' --opencl_binary_path %s' % (self.opencl_binary_path)
            run_line += ' --thread %d' % (self.thread)
        else:
            run_line += ' --cpu %d' % (self.num_cpu_dock)
            run_line += ' --exhaustiveness %d' % (self.exhaustiveness)
        # run_line += ' --cpu %d' % (self.num_cpu_dock)
        run_line += ' --num_modes %d ' % (self.num_modes)
        # print("Docking command:")
        # print(run_line)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        # print(result)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def docking_task(self, idx, smi):
        receptor_file = self.receptor_file
        ligand_mol_file = f'{self.temp_dir}/ligand_{idx}.mol'
        ligand_pdbqt_file = f'{self.temp_dir}/ligand_{idx}.pdbqt'
        docking_pdbqt_file = f'{self.temp_dir}/dock_{idx}.pdbqt'

        try:
            self.gen_3d(smi, ligand_mol_file)
        except Exception as e:
            print(f"gen_3d error for index {idx}")
            return idx, 99.9

        try:
            affinity_list = self.docking(receptor_file, ligand_mol_file,
                                         ligand_pdbqt_file, docking_pdbqt_file)
        except Exception as e:
            print(f"Docking error for index {idx}, error: {e}")
            return idx, 99.9

        if len(affinity_list) == 0:
            print(f"affinity_list error for index {idx}")
            return idx, 99.9

        affinity = affinity_list[0]
        return idx, affinity
    
    def docking_parallel(self, receptor_file, expected_num):
        assert self.use_gpu and self.gpu_parallel, "gpu_parallel is only supported in gpu mode."
        # ms = list(pybel.readfile("mol", ligand_mol_file))
        # m = ms[0]
        # m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        run_line = '%s --receptor %s --ligand_directory %s --output_directory %s' % (self.vina_program,
                                                              receptor_file, self.temp_dir, self.temp_dir)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' % (self.box_size)
        run_line += ' --opencl_binary_path %s' % (self.opencl_binary_path)
        run_line += ' --thread %d' % (self.thread)
        run_line += ' --num_modes %d ' % (self.num_modes)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        # print(result)
        result_lines = result.split('\n')

        check_result = False
        affinity_array = np.array([99.9] * expected_num)
        collect_num = 0
        for result_line in result_lines:
            if result_line.startswith("Refining ligand"):
                ligand_index = result_line.split("ligand_")[1].split(" results")[0]
                continue
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                # break
                check_result = False
                continue
            affinity = float(lis[1])
            if lis[0] == "1": ## only pick the first/best result
                # affinity_list += [affinity]
                affinity_array[int(ligand_index)] = affinity
                collect_num += 1
        if collect_num != expected_num:
            print(f"Warning: expected {expected_num} results, but got {collect_num}.")
        return affinity_array

    # Parallelize 3D generation using multiprocessing
    def gen_3d_task(self, idx_smi):
        idx, smi = idx_smi
        ligand_mol_file = f'{self.temp_dir}/ligand_{idx}.mol'
        ligand_pdbqt_file = f'{self.temp_dir}/ligand_{idx}.pdbqt'
        try:
            self.gen_3d(smi, ligand_mol_file)
            ms = list(pybel.readfile("mol", ligand_mol_file))
            m = ms[0]
            m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
            return idx, True
        except Exception as e:
            print(f"[Warning] gen_3d error for index {idx}")
            return idx, False
            
    def docking_parallel_tasks(self, smiles_list):
        receptor_file = self.receptor_file
        valid_indices = []
        affinity_array = np.array([99.9] * len(smiles_list))
        # for idx, smi in enumerate(smiles_list):
        #     ligand_mol_file = f'{self.temp_dir}/ligand_{idx}.mol'
        #     ligand_pdbqt_file = f'{self.temp_dir}/ligand_{idx}.pdbqt'
        #     # docking_pdbqt_file = f'{self.temp_dir}/dock_{idx}.pdbqt'

        #     try:
        #         self.gen_3d(smi, ligand_mol_file)
        #         ## TODO: debug whether this is necessary
        #         ms = list(pybel.readfile("mol", ligand_mol_file))
        #         m = ms[0]
        #         m.write("pdbqt", ligand_pdbqt_file, overwrite=True)
        #         valid_indices.append(idx)
        #     except Exception as e:
        #         print(f"[Warning] gen_3d error for index {idx}")
        #         # return idx, 99.9

        # Parallel 3D generation
        parallel = True # False
        # print("Parallel: ", parallel)
        if parallel:
            with mp.Pool(processes=20) as pool:
                results = pool.map(self.gen_3d_task, enumerate(smiles_list))
            
            valid_indices = [idx for idx, success in results if success]
        else:
            print("Generating 3D structures sequentially...")
            valid_indices = []
            for idx, smi in enumerate(smiles_list):
                _, tag_ = self.gen_3d_task((idx, smi))
                if tag_:
                    valid_indices.append(idx)

        max_trial_times = 3
        fail_times = 0
        while fail_times < max_trial_times:
            try:
                valid_affinity_list = self.docking_parallel(receptor_file, len(valid_indices))
                affinity_array[valid_indices] = valid_affinity_list
                break
            except Exception as e:
                fail_times += 1
                print(f"Docking error: error: {e} Failed {fail_times} times")
        if fail_times == max_trial_times:
            print(f"Docking failed for {len(valid_indices)} samples after {max_trial_times} attempts. Returning default values.")
                # return idx, 99.9
        return list(affinity_array)
        # if len(affinity_list) == 0:
        #     print(f"affinity_list error for index {idx}")
        #     return idx, 99.9

        # affinity = affinity_list[0]
        # return idx, affinity

    def predict(self, smiles_list):
        # Prepare data for parallel processing
       

        if not self.gpu_parallel or not self.use_gpu:
            tasks = [(idx, smi) for idx, smi in enumerate(smiles_list)]
            pool = mp.Pool(processes=self.num_sub_proc) # mp.cpu_count()) #len(tasks))  # Create a pool with num_sub_proc processes
            # num_cores = mp.cpu_count()  # Get the total number of CPU cores available
            # pool = mp.Pool(processes=self.num_sub_proc)  # Create a pool with num_sub_proc processes
            # Run docking in parallel
            results = pool.starmap(self.docking_task, tasks)

            pool.close()
            pool.join()

            # Collect results
            results.sort(key=lambda x: x[0])  # maintain order
            affinity_list = [result[1] for result in results]
            if os.path.exists(self.temp_dir):
                rmtree(self.temp_dir)
                print(f'{self.temp_dir} removed.')
        else:
            affinity_list = []
            batch_num = int(np.ceil(len(smiles_list) / self.eval_batch_size))
            for i in range(batch_num):
                # if i > 0:
                #     self.temp_dir = make_docking_dir(self.base_temp_dir)
                #     # if not os.path.exists(self.temp_dir):
                #     #     os.makedirs(self.temp_dir)
                batch_smiles = smiles_list[i * self.eval_batch_size:(i + 1) * self.eval_batch_size]
                if batch_num > 1 and len(batch_smiles) < self.eval_batch_size:
                    print(f"Warning: batch {i} has only {len(batch_smiles)} smiles, which is less than eval_batch_size {self.eval_batch_size}.")
                    rmtree(self.temp_dir)
                    self.temp_dir = make_docking_dir(self.base_temp_dir)

                batch_affinities = self.docking_parallel_tasks(batch_smiles)
                affinity_list.extend(batch_affinities)
            if os.path.exists(self.temp_dir):
                rmtree(self.temp_dir)
                print(f'{self.temp_dir} removed.')

        return affinity_list
    

############################ NoT Used ############################
    def docking_tmp(self, receptor_file, ligand_pdbqt_file, docking_pdbqt_file):
        run_line = '%s --receptor %s --ligand %s --out %s' % (self.vina_program,
                                                              receptor_file, ligand_pdbqt_file, docking_pdbqt_file)
        run_line += ' --center_x %s --center_y %s --center_z %s' % (self.box_center)
        run_line += ' --size_x %s --size_y %s --size_z %s' % (self.box_size)
        if self.use_gpu:
            run_line += ' --opencl_binary_path %s' % (self.opencl_binary_path)
            run_line += ' --thread %d' % (self.thread)
        else:
            run_line += ' --cpu %d' % (self.num_cpu_dock)
            run_line += ' --exhaustiveness %d' % (self.exhaustiveness)
        run_line += ' --num_modes %d ' % (self.num_modes)
        # run_line += ' --cpu %d' % (self.num_cpu_dock)
        # run_line += ' --num_modes %d' % (self.num_modes)
        # run_line += ' --exhaustiveness %d ' % (self.exhaustiveness)
        result = subprocess.check_output(run_line.split(),
                                         stderr=subprocess.STDOUT,
                                         timeout=self.timeout_dock, universal_newlines=True)
        result_lines = result.split('\n')

        check_result = False
        affinity_list = list()
        for result_line in result_lines:
            if result_line.startswith('-----+'):
                check_result = True
                continue
            if not check_result:
                continue
            if result_line.startswith('Writing output'):
                break
            if result_line.startswith('Refine time'):
                break
            lis = result_line.strip().split()
            if not lis[0].isdigit():
                break
            affinity = float(lis[1])
            affinity_list += [affinity]
        return affinity_list

    def creator(self, q, data, num_sub_proc):
        for d in data:
            idx = d[0]
            dd = d[1]
            q.put((idx, dd))

        for i in range(0, num_sub_proc):
            q.put('DONE')

    def docking_subprocess(self, q, return_dict, sub_id=0):
        while True:
            qqq = q.get()
            if qqq == 'DONE':
                break
            (idx, smi) = qqq
            # print(smi)
            receptor_file = self.receptor_file
            ligand_mol_file = '%s/ligand_%s.mol' % (self.temp_dir, sub_id)
            ligand_pdbqt_file = '%s/ligand_%s.pdbqt' % (self.temp_dir, sub_id)
            docking_pdbqt_file = '%s/dock_%s.pdbqt' % (self.temp_dir, sub_id)
            try:
                self.gen_3d(smi, ligand_mol_file)
            except Exception as e:
                print("gen_3d error")
                return_dict[idx] = 99.9
                continue
            try:
                affinity_list = self.docking(receptor_file, ligand_mol_file,
                                             ligand_pdbqt_file, docking_pdbqt_file)
            except Exception as e:
                print("Docking error")
                return_dict[idx] = 99.9
                continue
            if len(affinity_list) == 0:
                print("affinity_list error")
                affinity_list.append(99.9)
            
            affinity = affinity_list[0]
            return_dict[idx] = affinity