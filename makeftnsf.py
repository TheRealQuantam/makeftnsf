# makeftnsf.py
# Copyright 2024 Justin Olbrantz (Quantam)

# Tool to generate a bhop NSF from a Mega Man 2 Randomizer FtSoundTrackConfiguration.xml file to verify tracks work properly in bhop and test CPU usage.

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

import base64 
import collections as colls
from ctypes import c_char, c_byte, c_int8, c_uint8, c_int16, c_uint16, c_uint32, c_int32, Array, Structure, BigEndianStructure, LittleEndianStructure, sizeof, addressof
from enum import Enum, IntEnum, IntFlag, auto
import itertools
import math
from pathlib import Path
import sys
import traceback
from typing import *
import xml.etree.ElementTree
import zlib

import famitrackerbinary as ftb

bank_size = 0x1000 # NSF mapper bank size
mod_bank_base = 0xa000
max_banks = 256 # NSF mapper banks
max_tracks = 255
max_mod_size = 0x2000

# Insert gap at the end of non-looping tracks
non_looping_gap = 1

# Play 2 loops if the song is short enough
two_loop_thresh = 90 # Sec
# 4 loops if the song is REALLY short
four_loop_thresh = 7

stub_path = Path("nsfstub/nsfstub.nsf")
xml_filename = "FtSoundTrackConfiguration.xml"
xml_path_strs = (
	xml_filename,
	"../../MM2RandoLib/Resources/" + xml_filename,
)

c_uint16_le = c_uint16.__ctype_le__
c_uint32_le = c_uint32.__ctype_le__

class NsfHeader(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("magic", c_char * 5),
		("version", c_uint8),
		("num_tracks", c_uint8),
		("start_track", c_uint8), # 1-based
		("load_addr", c_uint16),
		("init_addr", c_uint16),
		("play_addr", c_uint16),
		("game_name", c_char * 32),
		("artist_name", c_char * 32),
		("copyright_name", c_char * 32),
		("ntsc_speed", c_uint16),
		("start_banks", c_uint8 * 8),
		("pal_speed", c_uint16),
		("machine_flags", c_uint8),
		("expansion_chips", c_uint8),
		("_reserved", c_uint8),
		("prg_size_lo", c_uint16),
		("prg_size_hi", c_uint8),
	)

ChunkId = c_char * 4

class NsfeChunkHdr(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("chunk_size", c_uint32), # Excludes header
		("chunk_id", ChunkId),
	)

class NsfeInfoChunk(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("load_addr", c_uint16),
		("init_addr", c_uint16),
		("play_addr", c_uint16),
		("machine_flags", c_uint8),
		("expansion_chips", c_uint8),
		("num_tracks", c_uint8),
		("start_track", c_uint8), # 0-based
	)

class Uses(IntFlag):
	Intro = auto()
	Title = auto()
	StageSelect = auto()
	Stage = auto()
	Boss = auto()
	Refights = auto()
	Credits = auto()

default_uses = Uses.Stage | Uses.Credits
usage_map = {usage.name.lower(): usage for usage in Uses}

class TrackEntry(NamedTuple):
	module: "ModuleEntry"
	index: int

	title: Optional[str] # None for auto-generated tracks

	enabled: Optional[bool]
	uses: Optional[Uses]

	swap_square_chans: Optional[bool]

class ModuleEntry(NamedTuple):
	title: str
	author: str

	enabled: bool
	uses: Uses

	base_addr: int
	data: bytes

	swap_square_chans: bool

	tracks: Dict[int, TrackEntry]

class TrackMapEntry(Structure):
	_fields_ = (
		("bank_idx", c_uint8),
		("track_idx", c_uint8)
	)

# Helpers to make it more convenient to assemble binary files
def append_struct(
	buff: bytearray, 
	struct: Union[Structure, Array],
) -> None:
	buff.extend((c_uint8 * sizeof(struct)).from_buffer(struct))

def append_chunk_hdr(
	buff: bytearray, 
	chunk_id: ChunkId, 
	size: int,
) -> None:
	append_struct(buff, NsfeChunkHdr(size, chunk_id))
	
def append_chunk(
	buff: bytearray, 
	chunk_id: ChunkId, 
	data: ByteString,
) -> None:
	append_chunk_hdr(buff, chunk_id, len(data))
	buff.extend(data)

def append_struct_chunk(
	buff: bytearray, 
	chunk_id: ChunkId, 
	struct: Structure,
) -> None:
	append_chunk_hdr(buff, chunk_id, sizeof(struct))
	append_struct(buff, struct)

def find_xml_file(cli_path: Optional[str]) -> Optional[Path]:
	if cli_path:
		xml_path = Path(cli_path)
		if xml_path.is_dir():
			xml_path = xml_path.joinpath(xml_filename)

	else:
		xml_path = None
		for path_str in xml_path_strs:
			xml_path = Path(path_str)
			if xml_path.is_file():
				break

	if not xml_path or not xml_path.is_file():
		print(f"'{xml_filename}' not found. Either run makeftnsf.py in the same directory as it or specify the path to it on the command line.")

		return None

	return xml_path

def parse_uses(uses_el: Optional[xml.etree.ElementTree.Element]) -> Optional[Uses]:
	"""Parse the relatively complex Uses element into a simple Uses enum."""

	if not uses_el:
		return None

	uses = Uses(0)
	for usage_el in uses_el.iter("Usage"):
		uses |= usage_map[usage_el.text.lower()]

	return uses

def load_mod_list(path: Path) -> Sequence[ModuleEntry]:
	"""Load the list of modules and tracks from the XML file."""

	tree = xml.etree.ElementTree.parse(path)
	root = tree.getroot()

	assert root.tag == "ModuleSet"

	mods = []
	for ste in root.find("Modules").iter("Module"):
		values = { el.tag: el.text for el in ste if not list(el) }
		
		try:
			# Parse most of the module
			uses = parse_uses(ste.find("Uses"))
			if uses is None:
				uses = default_uses

			data_str = values.get("Data")
			if data_str is not None:
				int_data = data_str.encode("ascii")
				
				if int_data.startswith(b"deflate:"):
					data = zlib.decompress(base64.standard_b64decode(data_str[8:]))
				else:
					data = base64.standard_b64decode(int_data)
				
			else:
				data = bytes.fromhex(values["TrackData"])
				
			tracks: Dict[int, TrackEntry] = colls.OrderedDict()
			mod = ModuleEntry(
				values["Title"],
				values.get("Author", ""),
				values.get("Enabled", "true").lower() == "true",
				uses,
				int(values.get("StartAddress", "0"), 16),
				data,
				values.get("SwapSquareChans", "false").lower() == "true",
				tracks,
			)

			# Parse the track list
			tracks_el = ste.find("Songs")
			if tracks_el:
				for track_el in tracks_el.iter("Song"):
					values = { el.tag: el.text for el in track_el if not list(el) }

					swap_square_chans = values.get("SwapSquareChans")
					swap_square_chans = (swap_square_chans.lower() == "true") if swap_square_chans is not None else None

					is_en = values.get("Enabled")
					if is_en is not None:
						is_en = is_en.lower() == "true"

					track = TrackEntry(
						mod,
						int(values["Number"]),
						values["Title"],
						is_en,
						parse_uses(track_el.find("Uses")),
						swap_square_chans,
					)
					tracks[track.index] = track

			else:
				# Create a TrackEntry for the module if no tracks are explicitly defined
				track = TrackEntry(mod, 0, None, None, None, None)
				tracks[0] = track

		except Exception as e:
			traceback.print_exc();

			print(f"\nERROR: {path}: {e}")

			continue

		mods.append(mod)

	return mods

def select_file_mods(mods: Sequence[ModuleEntry], mod_idx: int, bank_idx: int) -> Tuple[int, Sequence[ModuleEntry]]:
	"""Select as many modules as will fit in one file's worth of space and track entries."""

	file_mods = []
	track_idx = 0

	while mod_idx < len(mods):
		mod = mods[mod_idx]

		assert len(mod.data) <= max_mod_size

		num_tracks = len(mod.tracks)
		num_banks = int(math.ceil(len(mod.data) / bank_size))
		if bank_idx + num_banks > max_banks or track_idx + num_tracks > max_tracks:
			break

		file_mods.append(mod)
		mod_idx += 1

		bank_idx += num_banks
		track_idx += num_tracks

	return mod_idx, file_mods

def append_banks(banks: List[ByteString], data: ByteString, start_offs: int = 0) -> int:
	"""Break up data into banks and append them to the bank list."""

	num_banks = 0
	for offs in range(start_offs, len(data), bank_size):
		end_offs = offs + bank_size
		pad_size = end_offs - min(end_offs, len(data))
		banks.append(data[offs:end_offs] + b"\0" * pad_size)

		num_banks += 1

	return num_banks

def import_file_mods(
	banks: List[ByteString], 
	file_mods: Sequence[ModuleEntry],
) -> Tuple[Sequence[TrackMapEntry], Sequence[Tuple[float, float]], Sequence[str], Sequence[str]]:
	"""Load the previously selected modules and pack them into banks."""

	track_map = []
	track_lens = []
	titles = []
	authors = []

	for mod in file_mods:
		ftm = ftb.Module(mod.data, mod.base_addr)
		assert not ftm.dpcm_size

		ftm.change_base_addr(mod_bank_base)

		for track in mod.tracks.values():
			assert track.index < len(ftm.songs)

			if (track.swap_square_chans if track.swap_square_chans is not None else mod.swap_square_chans):
				ftm.swap_song_square_chans(track.index)

			track_map.append(TrackMapEntry(len(banks), track.index))

			# Lots of work to assemble the title
			title_parts = [mod.title]
			if track.title or track.index:
				title_parts.append(f"- ({track.index})")
				if track.title:
					title_parts.append(track.title)

			uses = track.uses if track.uses is not None else mod.uses
			uses_parts: List[str] = []
			uses_str = ", ".join((usage.name for usage in Uses if uses & usage))
			title_parts.append(f"({uses_str})")

			if not (track.enabled if track.enabled is not None else mod.enabled):
				title_parts.append("(DISABLED)")
					
			titles.append(" ".join(title_parts))
			authors.append(mod.author or "")

			track_lens.append(ftm.get_song_length(track.index))

		data = ftm.binary
		append_banks(banks, data)

	return track_map, track_lens, titles, authors

def write_nsf(
	path: Path, 
	hdr: NsfHeader, 
	banks: Sequence[ByteString],
	modules: Sequence[ModuleEntry], 
	**kwargs,
) -> None:
	# Build the file, starting with the header
	num_tracks = len(list(itertools.chain.from_iterable((mod.tracks for mod in modules))))

	rom = bytearray()
	hdr = NsfHeader.from_buffer_copy(hdr)
	hdr.num_tracks = num_tracks

	for key in ("game_name", "artist_name", "copyright_name"):
		value = kwargs.get(key)
		if value:
			setattr(hdr, key, value.encode("ascii"))

	append_struct(rom, hdr)

	for bank in banks:
		rom.extend(bank)

	path.write_bytes(rom)

	return

def write_nsfe(
	path: Path, 
	nsf_hdr: NsfHeader, 
	banks: Sequence[ByteString],
	modules: Sequence[ModuleEntry], 
	track_lens_ms: Sequence[int],
	fade_lens_ms: Sequence[int],
	titles: Sequence[str],
	authors: Sequence[str],
	**kwargs,
) -> None:
	# Start with the headers
	num_tracks = len(list(itertools.chain.from_iterable((mod.tracks for mod in modules))))

	rom = bytearray(b"NSFE")

	info = NsfeInfoChunk(
		nsf_hdr.load_addr,
		nsf_hdr.init_addr,
		nsf_hdr.play_addr,
		nsf_hdr.machine_flags,
		nsf_hdr.expansion_chips,
		num_tracks,
		nsf_hdr.start_track - 1, # Different bases
	)
	append_struct_chunk(rom, b"INFO", info)

	append_chunk(rom, b"BANK", nsf_hdr.start_banks)

	# Then the bank data. This is done so the track table and Play function will always be at a known location.
	append_chunk_hdr(rom, b"DATA", len(banks) * bank_size)
	base_offs = len(rom)
	for bank in banks:
		rom.extend(bank)

	# Then the track metadata
	lens_chunk = (c_uint32_le * num_tracks)(*track_lens_ms)
	append_struct_chunk(rom, b"time", lens_chunk)

	fade_chunk = (c_uint32_le * num_tracks)(*fade_lens_ms)
	append_struct_chunk(rom, b"fade", fade_chunk)

	label_chunk = "\0".join(titles + [""]).encode("utf-8")
	append_chunk(rom, b"tlbl", label_chunk)

	author_chunk = "\0".join(authors + [""]).encode("utf-8")
	append_chunk(rom, b"taut", author_chunk)

	# And NSF metadata
	names_chunk = "\0".join(
		[
			kwargs.get(f"{key}_name", "") 
			for key in "game artist copyright ripper".split()
		] + [""]
	).encode("utf-8")
	if len(names_chunk) > 4:
		append_chunk(rom, b"auth", names_chunk)

	append_chunk_hdr(rom, b"NEND", 0)

	path.write_bytes(rom)

	return

def main() -> None:
	# Parse the list of modules and tracks
	xml_path = find_xml_file(sys.argv[1] if len(sys.argv) > 1 else None)
	if not xml_path:
		return

	out_name = sys.argv[2] if len(sys.argv) > 2 else xml_path.stem

	mods = load_mod_list(xml_path)

	# Load the stub
	stub_rom = stub_path.read_bytes()
	hdr = NsfHeader.from_buffer_copy(stub_rom)

	# Split up the modules if there are more than can fit in one NSF
	file_idx = 0
	mod_idx = 0
	while mod_idx < len(mods):
		banks: List[ByteString] = []
		append_banks(banks, stub_rom[sizeof(hdr):])

		# Select and import the modules
		mod_idx, file_mods = select_file_mods(mods, mod_idx, len(banks))
		if not file_mods:
			break

		track_map, track_lens, titles, authors = import_file_mods(banks, file_mods)

		# Set up the track map
		rom_bank_0 = banks[0] = bytearray(banks[0])
		rom_track_map = (TrackMapEntry * len(track_map)).from_buffer(rom_bank_0)
		rom_track_map[:] = track_map[:]

		# Write the NSF
		num_str = str(file_idx) if file_idx or mod_idx < len(mods) else ""
		out_path = Path(f"{out_name}{num_str}.nsf")
		write_nsf(out_path, hdr, banks, file_mods)

		# Write the NSFe
		track_lens_ms = []
		fade_lens_ms = []
		for intro_len, loop_len in track_lens:
			total_len = intro_len + loop_len
			if loop_len > 0:
				if total_len / 60 <= four_loop_thresh:
					total_len += loop_len * 3
				elif total_len / 60 <= two_loop_thresh:
					total_len += loop_len
					
			else:
				total_len += non_looping_gap * 60

			track_lens_ms.append(int(math.ceil(total_len * 1000 / 60)))
			fade_lens_ms.append(15000 if loop_len else 0)

		out_path = out_path.with_suffix(".nsfe")
		write_nsfe(out_path, hdr, banks, file_mods, track_lens_ms, fade_lens_ms, titles, authors)
		
		file_idx += 1

	return

main()
