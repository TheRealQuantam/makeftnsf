# ftexporter.py
# Copyright 2024 Justin Olbrantz (Quantam)

# Interface to the FamiTracker EXE to export BIN files from FTM files.

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import famitrackerbinary as ftbin
import os
import re
import subprocess
import tempfile

from common import *

ft_exe_names = "Dn-FamiTracker.exe".split()

_ExChips = ftbin.ExpansionChips

class ExportFailed(RuntimeError):
	pass

class FamiTrackerExporter:
	def __init__(
		self, 
		path: Optional[Path] = None, 
		exe_names: Iterable[str] = ft_exe_names,
	) -> None:
		if path:
			if path.is_file():
				self._exe_path = path

				logging.debug(f" Found FamiTracker at '{self._exe_path}'")
			else:
				raise FileNotFoundError(path)

		else:
			paths = os.get_exec_path()
			#logging.debug(f" Path: {paths}")

			for path_str in ["."] + paths:
				path = Path(path_str)
				for exe_name in exe_names:
					exe_path = path.joinpath(exe_name)
					if exe_path.is_file():
						self._exe_path = exe_path.resolve()

						logging.debug(f" Found FamiTracker at '{self._exe_path}'")

						return

			raise FileNotFoundError(exe_names[0])

	def export_bin(
		self, 
		path: Path, 
		temp_path: Path,
	) -> None:
		bin_path = Path(tempfile.mktemp(".bin", "mod", dir = temp_path))
		dpcm_path = Path(tempfile.mktemp(".bin", "dpcm", dir = temp_path))
		log_path = Path(tempfile.mktemp(".txt", "log", dir = temp_path))

		res = subprocess.run(
			(str(self._exe_path), str(path), "-export", str(bin_path), str(log_path), str(dpcm_path)),
			input = b"\r\n" * 5,
			capture_output = True,
			check = False,
		)

		log = log_path.read_text()
		match = re.search(r"^ Error: \s* ( .* ) $", log, re.X | re.M)
		if res.returncode or match:
			raise ExportFailed(match[1] if match else "Export failed", res.returncode)

		assert bin_path.exists()

		if "No expansion chip" in log:
			ex_chips = _ExChips(0)
		elif "Multiple expansion chips" in log:
			ex_chips = _ExChips(-1)
		else:
			ex_chips = _ExChips(0)
			for chip in _ExChips:
				if f"{chip.name} expansion" in log:
					ex_chips |= chip

			assert ex_chips in _ExChips

		song_nums = set((int(idx) for idx in re.findall(r"^ \s* \* \s* Song \s+ ( \d+ ) :", log, re.I | re.M | re.X)))
		assert max(song_nums) + 1 == len(song_nums)
	 
		match = re.search(r"Samples located at: \$([a-fA-F\d]{4})", log)
		dpcm_base = int(match[1], 16) if match else None

		if dpcm_path.exists():
			dpcm_path = dpcm_path.resolve()
			dpcm_size = dpcm_path.stat().st_size

			assert not dpcm_size or dpcm_base is not None

		else:
			dpcm_path = dpcm_size = None

		bin_path = bin_path.resolve()
		size = bin_path.stat().st_size
		num_songs = len(song_nums)

		logging.debug(f" Exported FT file '{path}': {num_songs} songs, ${size:x} bytes data, ${dpcm_size or 0:x} bytes samples")

		return {
			"bin_path": bin_path,
			"bin_size": size,
			"num_tracks": num_songs,
			"dpcm_path": dpcm_path if dpcm_size else None,
			"dpcm_base": dpcm_base,
			"dpcm_size": dpcm_size,
			#"has_ex_chips": ex_chips != 0,
			"ex_chips": ex_chips,
		}

	def export_text(
		self, 
		src_path: Path, 
		temp_path: Path,
	) -> Path:
		tgt_path = Path(tempfile.mktemp(".txt", "mod", dir = temp_path))
		log_path = Path(tempfile.mktemp(".txt", "log", dir = temp_path))
		res = subprocess.run(
			(str(self._exe_path), str(src_path), "-export", str(tgt_path), str(log_path)),
			input = b"\r\n" * 5,
			capture_output = True,
			check = False,
		)

		log = log_path.read_text()
		if res.returncode:
			match = re.search(r"^ Error: \s* ( .* ) $", log, re.X)
			raise ExportFailed(match[1] if match else "Export failed", res.returncode)

		assert tgt_path.is_file()

		return tgt_path

	def import_txt(
		self,
		src_path: Path,
		temp_path: Path,
	) -> Path:
		tgt_path = Path(tempfile.mktemp(".dnm", "mod", dir = temp_path))
		res = subprocess.run(
			(str(self._exe_path), str(src_path), "-convert", str(tgt_path)),
			input = b"\r\n" * 5,
			capture_output = True,
			check = False,
		)

		if res.returncode:
			match = re.search(r"^ Error: \s* ( .* ) $", res.stderr, re.X)
			raise ExportFailed(match[1] if match else "Export failed", res.returncode)

		assert tgt_path.is_file()

		return tgt_path
