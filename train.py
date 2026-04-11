from run import DLF_run

def main():
    DLF_run(
        model_name='STUDENT',
        dataset_name='mosi',
        is_tune=False,
        seeds=[1111],
        res_save_dir="./result",
        log_dir="./log",
        mode='train',#切换到test，这后面两个参数都要改。
        is_training=True
    )

if __name__ == '__main__':
    main()