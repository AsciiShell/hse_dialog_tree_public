import heapq

from tqdm import tqdm

from hse_dialog_tree.utils.files import dump_gzip_text, load_gzip_text_iter


def sort_merged_files(paths, output_file, output_size=1_000_000, dedup=True):
    handles = [load_gzip_text_iter(path) for path in paths]

    output_data = []
    output_part_id = 0
    last_line = None
    for line in tqdm(heapq.merge(*handles), total=136_000_000):
        if not dedup or line != last_line:
            last_line = line
            output_data.append(line)
        if len(output_data) >= output_size:
            dump_gzip_text(output_data, output_file.format(output_part_id))
            output_part_id += 1
            output_data = []
    dump_gzip_text(output_data, output_file.format(output_part_id))
