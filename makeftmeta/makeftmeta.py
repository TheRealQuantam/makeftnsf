# makeftmeta.py
# Copyright 2024 Justin Olbrantz (Quantam)

# Tool to auto-generate metadata for FTMs that can be used with makeftnsf, various FT-based randomizers, etc.

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.
 
import base64
import collections as colls
import concurrent.futures as futures
from ctypes import c_char, c_byte, c_int8, c_uint8, c_int16, c_uint16, c_uint32, c_int32, Array, Structure, BigEndianStructure, LittleEndianStructure, sizeof, addressof
from enum import Enum, IntEnum, IntFlag, auto
import glob
import itertools
import math
from pathlib import Path
import re
import sys
import tempfile
import traceback
from typing import *
import xml.dom.minidom
import xml.etree.ElementTree as etree
import xml.sax.saxutils as saxutils
import zlib

import famitrackerbinary as ftb
import ftexporter

class FtmEntry(NamedTuple):
	src_path: Path

	info: Dict[str, Any]
	data: bytes
	module: ftb.Module

	text: str
	strings: Dict[str, str]
	song_names: List[str]

def export_mod(
	exporter: ftexporter.FamiTrackerExporter,
	src_path: Path,
	tmp_dir: tempfile.TemporaryDirectory,
) -> None:
	print(f"Processing '{src_path}'...")

	try:
		exp_info = exporter.export_bin(src_path, tmp_dir)
	except ftexporter.ExportFailed as e:
		if "bankswitched" in e.args[0]:
			print(f"FTM '{src_path}' is too big: {e.args[0]}. Skipping.")
			return
		
		raise

	bin_size = exp_info["bin_size"]
	if bin_size > max_size:
		print(f"FTM '{src_path}' is too big ({bin_size} bytes). Skipping.")
		return

	if not (exp_info["dpcm_size"] <= max_dpcm_size):
		return

	data = exp_info["bin_path"].read_bytes()
	ftm = ftb.Module(data, dpcm_base_addr = exp_info["dpcm_base"])

	assert ftm.dpcm_size == exp_info["dpcm_size"]

	txt_path = exporter.export_text(src_path, tmp_dir)
	txt = txt_path.read_text()

	mod_info = {}
	for key, value in re.findall(r'^ \s* ( title | author | copyright ) \s* " ( .* ) " \s* $', txt, re.I | re.X | re.M):
		mod_info[key.lower()] = value

	song_names = []
	for name in re.findall(r'^ track \s* \d+ \s* \d+ \s* \d+ \s* " ( .* ) " \s* $', txt, re.I | re.X | re.M):
		song_names.append(name)


	ftms[src_path] = FtmEntry(src_path, exp_info, data, ftm, txt, mod_info, song_names)
	
max_size = 0x2000
max_dpcm_size = 0
ft_path = Path(r"H:\Downloads\Dn-FamiTracker-src\x64\Debug\Dn-FamiTracker.exe")
base_path = Path(sys.argv[2])
tgt_path = Path("out.xml")
allow_dpcm = False

fmt_str = sys.argv[1]
src_paths = []
for raw_str in sys.argv[3:]:
	for path_str in glob.iglob(raw_str, root_dir = base_path):
		path = base_path.joinpath(path_str)
		if path.is_file():
			src_paths.append(path)

assert src_paths

exporter = ftexporter.FamiTrackerExporter(ft_path)

max_threads = 8
thread_pool = futures.ThreadPoolExecutor(max_threads)
work_queue: Deque[futures.Future] = colls.deque()

ftms: Dict[Path, Optional[FtmEntry]] = {}
with tempfile.TemporaryDirectory(prefix = "makeftmeta") as tmp_dir:
	# Add to the dict first to preserve order
	for src_path in src_paths:
		if src_path in ftms:
			continue
		
		ftms[src_path] = None
		
	for src_path in ftms:
		while len(work_queue) >= max_threads:
			futr = work_queue.popleft()
			futr.result()
			
		futr = thread_pool.submit(export_mod, exporter, src_path, tmp_dir)
		work_queue.append(futr)
		
	while work_queue:
		futr = work_queue.popleft()
		futr.result()

deflate = False
if fmt_str == "xml-deflate":
	fmt_str = "xml"
	deflate = True
	
if fmt_str == "xml":
	SubElement = etree.SubElement
	root = etree.Element("ModuleSet")
	tree = etree.ElementTree(root)
	mods = SubElement(root, "Modules")

	for ftm_entry in filter(bool, ftms.values()):
		ftm = ftm_entry.module
		strings = ftm_entry.strings

		mod = SubElement(mods, "Module")
		SubElement(mod, "Enabled").text = "false"

		song_name = strings.get("title")
		mod_title = SubElement(mod, "Title")

		mod_name = song_name if song_name else str(ftm_entry.src_path.name)
		mod_title.text = (mod_name)

		author = strings.get("author")
		if author:
			SubElement(mod, "Author").text = (author)

		song_els = []
		for idx, song in enumerate(ftm.songs):
			song_el = etree.Element("Song")
			#SubElement(song_el, "Enabled").text = "true"
			SubElement(song_el, "Number").text = str(idx)

			song_name = ftm_entry.song_names[idx]
			SubElement(song_el, "Title").text = (song_name or f"Song {idx}")

			song_els.append(song_el)

		if len(song_els) > 1:
			songs_el = SubElement(mod, "Songs")
			songs_el.extend(song_els)

		elif len(song_els) == 1:
			if song_name:
				mod_title.text = (mod_name +  f" - {song_name}")

		if deflate:
			cmp_data = zlib.compress(ftm_entry.data, zlib.Z_BEST_COMPRESSION)
			data_text = "deflate:" + base64.standard_b64encode(cmp_data).decode(encoding = "ascii")
			
			SubElement(mod, "Data").text = data_text
		else:
			data_text = ftm_entry.data.hex()
			
			SubElement(mod, "TrackData").text = data_text

	xml_str = etree.tostring(root, "unicode", xml_declaration = True, short_empty_elements = False)
	dom = xml.dom.minidom.parseString(xml_str)
	tgt_path.write_bytes(dom.toprettyxml(encoding = "utf-8"))

a = 0