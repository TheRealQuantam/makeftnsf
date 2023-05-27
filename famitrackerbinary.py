# famitrackerbinary.py
# Copyright 2023 Justin Olbrantz (Quantam)

# Module to parse a Dn-FamiTracker BIN export and rebase it.

# This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License. To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

import collections as colls
import ctypes
from ctypes import c_char, c_byte, c_int8, c_uint8, c_int16, c_uint16, Structure, BigEndianStructure, LittleEndianStructure, sizeof, addressof
from enum import Enum, IntEnum, IntFlag, auto
import itertools
import operator
from typing import *

c_uint16_be = c_uint16.__ctype_be__
c_uint16_le = c_uint16.__ctype_le__

class Error(Exception):
	pass

class DataError(Error):
	def __init__(
		self, 
		err_str: str, 
		*args, 
		source: Optional[Any], 
		offset: Optional[int], 
		struct: Optional[type] = None, 
		field_name: Optional[str] = None, 
		value: Optional[Any] = None, 
		**kwargs,
	) -> None:
		self.xargs = {
			"err_str": err_str,
			"source": source,
			"offset": offset,
			"struct": struct,
			"field_name": field_name,
			"value": value,
		}
		self.xargs.update(kwargs)

		super().__init__(self, err_str, args, self.xargs)

	@staticmethod
	def from_field(
		err_str: str, 
		source: Optional[Any], 
		struct: Any, 
		field_name: str,
	) -> "DataError":
		offs = None
		if source is not None:
			try:
				base_addr = addressof(source)
			except TypeError:
				base_addr = addressof(c_byte.from_buffer(source))

			offs = addressof(struct) - base_addr

		return DataError(
			err_str, 
			source = source, 
			offset = offs,
			struct = struct,
			field_name = field_name,
			value = getattr(struct, field_name),
		)

class FormatError(DataError):
	pass

class UnsupportedError(DataError):
	pass

class ExpansionChips(IntFlag):
	VRC6 = 1
	VRC7 = 2
	FDS = 4
	MMC5 = 8
	N163 = 16
	S5B = 32

class Header(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("song_list", c_uint16),
		("instrument_list", c_uint16),
		("sample_list", c_uint16),
		("samples", c_uint16),
		("groove_list", c_uint16),
		("flags", c_uint8),
		("ntsc_speed", c_uint16), # Engine speed in frames/min
		("pal_speed", c_uint16),
	)

# Flags for Header.flags
class HeaderFlags(IntEnum):
	NoFlags = 0
	BankSwitched = 1 << 0
	OldVibrato = 1 << 1
	LinearPitch = 1 << 2

class SongInfo(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("frames", c_uint16),
		("frame_count", c_uint8),
		("pattern_length", c_uint8),
		("speed", c_uint8), # 0 if using groove_pos
		("tempo", c_uint8),
		("groove_pos", c_uint8),
		("bank", c_uint8),
	)

class DpcmInstrument(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("pitch", c_uint8),
		("unk1", c_uint8),
		("sample_idx", c_uint8),
	)

class Sample(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("address", c_uint8),
		("size", c_uint8),
		("bank", c_uint8),
	)

class InstrumentHeader(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("type", c_uint8),
		("env_mask", c_uint8),
		# Then an array of pointers to envelopes
	)

class InstrumentTypes(IntEnum):
	APU = 0
	Triangle = 1 # Not used
	Noise = 2 # Not used
	DPCM = 3 # Not used
	VRC6 = 4
	Sawtooth = 5 # Not used
	VRC7 = 6
	FDS = 7
	MMC5 = 8 # Not used
	N163 = 9
	S5B = 10

class Song(NamedTuple):
	info: SongInfo
	frame_addrs: List[int]
	frames: List[List[int]]

class Instrument(NamedTuple):
	info: InstrumentHeader
	seq_addrs: List[int]

_ExChips = ExpansionChips
num_ex_chip_chans = {
	_ExChips(0): 5,
	_ExChips.VRC6: 3,
	_ExChips.VRC7: 6,
	_ExChips.FDS: 1,
	_ExChips.MMC5: 2,
	#_ExChips.N163: 8,
	_ExChips.S5B: 3,
}

_mappable_ex_chip_chans = {
	_ExChips(0): [0, 1],
	_ExChips.VRC6: [0, 1],
	_ExChips.VRC7: [0, 1],
	_ExChips.FDS: [0, 1],
	_ExChips.MMC5: [0, 1, 4, 5],
	_ExChips.N163: [0, 1],
	_ExChips.S5B: [0, 1],
}

def _get_num_chans(chips: ExpansionChips) -> int:
	"""Get the total number of channels for an expansion chip combination."""

	return (num_ex_chip_chans[_ExChips(0)] + sum((
		num_ex_chip_chans[chip]
		for chip in _ExChips
		if chip & chips
	)))

class _SpeedTempoChange(NamedTuple):
	row_idx: int

	is_tempo: bool
	value: int

class _PatternInfo:
	def __init__(
		self, 
		num_rows: int, 
		tempo_changes: Sequence[_SpeedTempoChange],
		next_order: Optional[int],
		next_row: Optional[int],
	):
		self.num_rows = num_rows

		self.tempo_changes = tempo_changes

		self.next_order = next_order
		self.next_row = next_row

class Module:
	"""An FT module in Dn-FamiTracker bytecode format."""

	def __init__(
		self, 
		data: ByteString, 
		base_addr: int = 0, 
		dpcm_base_addr: Optional[int] = None,
	) -> None:
		data = self._data = bytearray(data)
		self._base_addr = base_addr
		self._dpcm_base_addr = dpcm_base_addr
		leca = self._leca = lambda addr: (addr - self._base_addr)

		self._load()

		return

	@property
	def binary(self) -> bytes:
		return bytes(self._data)

	@property
	def header(self):
		return self._hdr

	@property
	def flags(self):
		return HeaderFlags(self._hdr.flags)

	@property
	def dpcm_size(self) -> int:
		return self._dpcm_size

	@property
	def sample_list(self):
		return self._sample_list

	@property
	def samples(self):
		return self._samples

	@property
	def instruments(self):
		return self._instrs

	@property
	def songs(self):
		return self._songs

	@property
	def grooves(self):
		return self._grooves

	def get_song_length(self, song_idx: int, immediate_stop: bool = True) -> Tuple[float, float]:
		len_an = SongLengthAnalyzer(immediate_stop)
		return len_an.get_length(self, song_idx)

	def change_ex_chips(
		self, 
		src_ex_chips: ExpansionChips, 
		tgt_ex_chips: ExpansionChips,
	) -> None:
		class FragInfo(NamedTuple):
			src_addr: int
			tgt_addr: int
			size: int

		assert not (src_ex_chips & ~tgt_ex_chips)

		leca = self._leca
		data = self._data
		hdr = self._hdr
		src_frame_size = _get_num_chans(src_ex_chips) * 2
		tgt_frame_size = _get_num_chans(tgt_ex_chips) * 2
		
		# Find all the fragments of the file
		tgt_addrs = set()
		frame_addrs = set()
		pat_addrs = set()

		for name, ty in Header._fields_[:5]:
			tgt_addrs.add(getattr(hdr, name))
		
		tgt_addrs.update(self._song_addrs)
		for song in self._songs:
			tgt_addrs.add(song.info.frames)
			frame_addrs.update(song.frame_addrs)
			for frame in song.frames:
				pat_addrs.update(frame)

		tgt_addrs.update(self._instr_addrs)
		for instr in self._instrs:
			tgt_addrs.update(instr.seq_addrs)

		# TODO: Maintain pattern order and don't merge all dummy patterns??
		# Relocate all the fragments to accomodate the larger frames
		DummyPats = ((c_uint8 * 2) * len(self._songs))

		sorted_addrs = sorted(
			tgt_addrs.union(frame_addrs, pat_addrs, {0, len(self._data)}))
		frags = {}
		offs = sizeof(DummyPats)
		a, b = itertools.tee(sorted_addrs)
		next(b, None)
		for addr, end_addr in zip(a, b):
			size = end_addr - addr
			frags[addr] = FragInfo(addr, offs, size)

			if addr in frame_addrs:
				assert size == src_frame_size

				size = tgt_frame_size

			offs += size

		# Correct target address
		hdr_size = frags[0].size
		frags[0] = FragInfo(0, 0, hdr_size)

		new_size = offs
		new_data = bytearray(new_size)
		dummy_pats = DummyPats.from_buffer(new_data, hdr_size)

		# Rewrite the addresses for everything except the frames
		def copy_frag(frag):
			src_addr = frag.src_addr
			tgt_addr = frag.tgt_addr
			new_data[tgt_addr:tgt_addr + frag.size] = (
				data[src_addr:src_addr + frag.size])

			return

		def map_addrs(addrs):
			for idx, addr in enumerate(addrs):
				frag = frags[addr]
				copy_frag(frag)

				addrs[idx] = frag.tgt_addr

			return 

		TgtFrame = c_uint16_le * (tgt_frame_size // 2)
		if src_ex_chips == 0 and tgt_ex_chips == _ExChips.MMC5:
			def copy_frame(frag, src_frame, tgt_frame, dummy_pat_addr):
				tgt_addrs = [frags[addr].tgt_addr for addr in src_frame]
				tgt_frame[:] = (
					tgt_addrs[0:4]
					+ [dummy_pat_addr] * 2
					+ tgt_addrs[4:]
				)


		for instr in self._instrs:
			map_addrs(instr.seq_addrs)
		map_addrs(self._instr_addrs)

		for pat_addr in pat_addrs:
			copy_frag(frags[pat_addr])

		#SrcFrame = c_uint16_le * (src_frame_size // 2)
		NewFrame = c_uint16_le * (tgt_frame_size // 2)
		"""for frame_addr in frame_addrs:
			frag = frags[frame_addr]

			src_frame = SrcFrame.from_buffer(data, frame_addr)
			tgt_frame = TgtFrame.from_buffer(new_data, frag.tgt_addr)
			#copy_frame(frag, src_frame, tgt_frame, dummy_pats"""

		dummy_pat_addr = hdr_size
		for dummy_pat, song in zip(dummy_pats, self._songs):
			dummy_pat[:] = (0, song.info.pattern_length - 1)

			for idx, (addr, frame) in enumerate(
				zip(song.frame_addrs, song.frames)):
				frag = frags[addr]
				new_frame = NewFrame.from_buffer(new_data, frag.tgt_addr)

				copy_frame(frag, frame, new_frame, dummy_pat_addr)
				song.frame_addrs[idx] = frag.tgt_addr

			frag = frags[song.info.frames]
			copy_frag(frag)
			song.info.frames = frag.tgt_addr

			dummy_pat_addr += sizeof(dummy_pat)
		map_addrs(self._song_addrs)

		for name, ty in Header._fields_[:5]:
			addr = getattr(hdr, name)
			frag = frags[addr]

			copy_frag(frag)
			setattr(hdr, name, frag.tgt_addr)
		copy_frag(frags[0])
		
		# Finally, to make things easy, reload the module
		self._data = new_data
		self._load()

		return

	def change_base_addr(
		self, 
		new_base: int, 
		new_dpcm_base: int = 0xc000,
	) -> None:
		if self._samples:
			if new_dpcm_base < 0xc000 or new_dpcm_base > 0xffc0:
				raise ValueError("change_base_addr new_dpcm_base must be between 0xc000 and 0xffc0")
			if new_dpcm_base % 0x40:
				raise ValueError("change_base_addr new_dpcm_base must be a multiple of 64")
			if new_base + self._dpcm_size > 0x10000:
				raise ValueError("change_base_addr new_base overflows address space")

		hdr = self._hdr
		delta = new_base - self._base_addr

		hdr.song_list += delta
		hdr.instrument_list += delta
		hdr.sample_list += delta
		hdr.samples += delta
		hdr.groove_list += delta

		for song_idx, song in enumerate(self._songs):
			self._song_addrs[song_idx] += delta
			song.info.frames += delta

			for frame_idx, chan_addrs in enumerate(song.frames):
				song.frame_addrs[frame_idx] += delta
					
				for chan_idx in range(len(chan_addrs)):
					chan_addrs[chan_idx] += delta

		for instr_idx, instr in enumerate(self._instrs):
			self._instr_addrs[instr_idx] += delta

			for seq_idx in range(len(instr.seq_addrs)):
				instr.seq_addrs[seq_idx] += delta

		self._base_addr = new_base
		self._leca = lambda addr: (addr - self._base_addr)

		if self._samples and self._dpcm_base_addr is not None:
			delta = (new_dpcm_base - self._dpcm_base_addr) // 0x40
			for sample in self._samples:
				sample.address += delta
			
		return

	def swap_song_square_chans(self, song_idx: int) -> None:
		for frame in self._songs[song_idx].frames:
			frame[:2] = frame[1::-1]

	def remap_song_channels(
		self, 
		song_idx: int,
		ex_chips: ExpansionChips,
		chan_map: Mapping[int, int],
	) -> None:
		if not chan_map:
			return

		song = self._songs[song_idx]
		if not song.frames:
			return

		num_chans = _get_num_chans(ex_chips)
		if len(song.frames[0]) != num_chans:
			# Out of order patterns confused the module loader. TODO: Fix.
			return

		map_chans = _mappable_ex_chip_chans[ex_chips]
		map_chan_idcs = set(range(len(map_chans)))
		
		bad_idcs = (set(chan_map.keys()).union(chan_map.values())
			- map_chan_idcs)
		if bad_idcs:
			raise ValueError(f"remap_song_channels channel indices must be between 0 and {len(map_chans)}")

		chan_map = dict(chan_map)
		unused_src_idcs = sorted(
			map_chan_idcs.difference(chan_map.keys()))
		unused_tgt_idcs = sorted(
			map_chan_idcs.difference(chan_map.values()))
		for src_idx, tgt_idx in zip(unused_src_idcs, unused_tgt_idcs):
			chan_map[src_idx] = tgt_idx

		assert len(set(chan_map.values())) == len(map_chans)

		rev_chan_map = list(range(num_chans))
		for src_idx, tgt_idx in chan_map.items():
			rev_chan_map[map_chans[tgt_idx]] = map_chans[src_idx]

		chan_mapper = operator.itemgetter(*rev_chan_map)
		for frame in song.frames:
			frame[:] = chan_mapper(frame)

		return

	def _check_addr(
		self, 
		addr: Optional[int], 
		size: int = 1, 
		*, 
		allow_null = False,
	) -> bool:
		if addr is not None:
			offs = self._leca(addr)
			return offs >= 0 and offs + size <= len(self._data)
		else:
			return allow_null

	def _load(self) -> None:
		data = self._data
		hdr = self._hdr = Header.from_buffer(data)

		prev_addr = 0
		for name, ty in Header._fields_[:5]:
			addr = getattr(hdr, name)
			if not self._check_addr(addr):
				raise FormatError.from_field("Invalid address", data, hdr, "name")
			if addr < prev_addr:
				raise UnsupportedError.from_field(
					"Unsupported section ordering", data, hdr, name)

		if hdr.flags & 0xfc1:
			raise UnsupportedError.from_field("Unsupported flag", data, hdr, "flags")

		self._load_songs()
		self._load_instrs()
		self._load_samples()
		self._load_grooves()

		return

	def _load_songs(self) -> None:
		leca = self._leca
		data = self._data
		hdr = self._hdr

		self._songs: List[Song] = []
		if hdr.song_list != hdr.instrument_list:
			song_tbl_offs = leca(hdr.song_list)
			first_song_addr = c_uint16_le.from_buffer(data, song_tbl_offs).value
			num_songs = (first_song_addr - hdr.song_list) // 2

			self._song_addrs = (c_uint16_le * num_songs).from_buffer(
				data, song_tbl_offs)

			for song_addr in self._song_addrs:
				info = SongInfo.from_buffer(data, leca(song_addr))
				
				frame_offs = leca(info.frames)
				#first_frame_addr = c_uint16_le.from_buffer(data, frame_offs).value
				frame_addrs = (c_uint16_le * info.frame_count).from_buffer(
					data, frame_offs)

				if info.frame_count > 1:
					frame_end = frame_addrs[1]
					num_chans = (frame_end - frame_addrs[0]) // 2
				else:
					addrs_offs = leca(frame_addrs[0])
					num_addrs = (len(data) - addrs_offs) // 2
					chan_addrs = (c_uint16_le * num_addrs).from_buffer(
						data, addrs_offs)
					num_chans = 1
					frame_end = chan_addrs[0]

					addrs_offs += 2

					while addrs_offs < frame_end:
						frame_end = min(chan_addrs[num_chans], frame_end)
						num_chans += 1
						addrs_offs += 2
					
				Frame = c_uint16_le * num_chans
				frames = []
				for frame_addr in frame_addrs:
					frame = Frame.from_buffer(data, leca(frame_addr))
					frames.append(frame)

				self._songs.append(Song(info, frame_addrs, frames))

		else:
			self._song_addrs = ()

		return

	def _load_samples(self) -> None:
		leca = self._leca
		data = self._data
		hdr = self._hdr

		num_instrs = (hdr.samples - hdr.sample_list) // sizeof(DpcmInstrument)
		num_samples = (hdr.groove_list - hdr.samples) // sizeof(Sample)

		self._sample_list = (DpcmInstrument * num_instrs).from_buffer(
			data, leca(hdr.sample_list))
		self._samples = (Sample * num_samples).from_buffer(
			data, leca(hdr.samples))

		base_dpcm_addr = 0x10000
		end_dpcm_addr = 0xc000
		for sample in self._samples:
			base_addr = sample.address * 0x40 + 0xc000
			if self._dpcm_base_addr is not None and base_addr < self._dpcm_base_addr:
				raise FormatError.from_field(
					"Invalid DPCM address", self._data, sample, "address")

			end_addr = base_addr + sample.size * 16 + 1

			base_dpcm_addr = min(base_dpcm_addr, base_addr)
			end_dpcm_addr = max(end_dpcm_addr, end_addr)

			if end_dpcm_addr > 0x10000:
				raise FormatError.from_field(
					"DPCM sample overflow address space", self._data, sample, "size")

		self._dpcm_size = 0
		if base_dpcm_addr < end_dpcm_addr:
			if self._dpcm_base_addr is None:
				self._dpcm_base_addr = base_dpcm_addr

			self._dpcm_size = end_dpcm_addr - self._dpcm_base_addr

		return

	def _load_instrs(self) -> None:
		leca = self._leca
		data = self._data
		hdr = self._hdr

		self._instrs: List[Instrument] = []
		if hdr.instrument_list != hdr.sample_list:
			instr_tbl_offs = leca(hdr.instrument_list)
			first_instr_addr = c_uint16_le.from_buffer(data, instr_tbl_offs).value
			num_instrs = (first_instr_addr - hdr.instrument_list) // 2

			self._instr_addrs = (c_uint16_le * num_instrs).from_buffer(
				data, instr_tbl_offs)

			for instr_idx, instr_addr in enumerate(self._instr_addrs):
				info = InstrumentHeader.from_buffer(data, leca(instr_addr))
				seq_tbl_addr = instr_addr + sizeof(InstrumentHeader)
				num_seqs = bin(info.env_mask).count("1")
				seq_addrs = (c_uint16_le * num_seqs).from_buffer(
					data, leca(seq_tbl_addr))

				self._instrs.append(Instrument(info, seq_addrs))

		else:
			self._instr_addrs = ()

		return

	def _load_grooves(self) -> None:
		self._grooves = []

		if not self._hdr.groove_list:
			return

		grooves_offs = self._leca(self._hdr.groove_list)
		data = self._data
		b = data[grooves_offs]
		if b:
			raise FormatError(
				"Invalid groove data",
				source = self._data,
				offset = grooves_offs,
				value = b,
			)

		offs = grooves_offs + 1
		end_offs = grooves_offs + 255
		while offs < end_offs:
			null_offs = data.find(b"0", offs, end_offs)
			if null_offs < 0:
				break

			b = data[null_offs + 1]
			if b != offs - grooves_offs:
				break

			self._grooves.append(data[offs:null_offs])
			offs = null_offs + 2

		return

class SongLengthAnalyzer:
	"""Calculate the intro and loop lengths for the song in frames. May be fractional if not 150 tempo."""

	def __init__(self, immediate_stop: bool = True) -> None:
		self._immediate_stop = immediate_stop

	def get_length(self, mod: Module, song_idx: int) -> Tuple[float, float]:
		song = mod._songs[song_idx]

		# Parse the patterns to get lengths and speeds
		pat_infos: Dict[int, _PatternInfo] = {}
		for frame_idx, pat_addrs in enumerate(song.frames):
			self._add_pat_info(mod, song, pat_addrs, pat_infos)

		return self._calc_length(song, pat_infos)

	_op_lens = (
		(1,) * 0x80 # Notes and note-like critters
		+ (
			2, 1, 2, 1, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2,
			2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
		) # Effects
		+ (2,) * 0x40 # More effects
		+ (1,) * 0x20 # Quick volume and instrument changes
	)

	def _hdl_set_dur(self):
		self._def_dur = self._param + 1

	def _hdl_reset_dur(self):
		self._def_dur = None

	def _hdl_speed(self):
		self._tempo_changes.append(_SpeedTempoChange(self._row_idx, False, self._param))

	def _hdl_tempo(self):
		self._tempo_changes.append(_SpeedTempoChange(self._row_idx, True, self._param))

	def _hdl_jump(self):
		self._new_order = self._param - 1
		self._new_row = 0

	def _hdl_skip(self):
		self._new_order = None
		self._new_row = self._param - 1

	def _hdl_halt(self):
		self._new_order = -1
		self._stop = True

	def _add_pat_info(self, mod: Module, song: Song, pat_addrs: Iterable[int], pat_infos: Dict[int, _PatternInfo]) -> None:
		for chan_idx, pat_addr in enumerate(pat_addrs):
			pat_info = pat_infos.get(pat_addr)
			if pat_info:
				continue

			self._tempo_changes: List[_SpeedTempoChange] = []
			offs = pat_addr - mod._base_addr
			self._row_idx = 0
			self._def_dur = None
			self._new_order = self._new_row = None
			self._stop = False

			while self._row_idx < song.info.pattern_length:
				b = mod._data[offs]
				next_offs = offs + self._op_lens[b]

				if b >= 0x80:
					# Command
					self._param: Optional[int] = mod._data[offs + 1] if offs + 1 < len(mod._data) else None
					hdlr = self._op_hdlrs.get(b)
					if hdlr:
						hdlr(self)

						# Whether stop is immediate or defers till end of the row differs between FT versions
						if self._stop and self._immediate_stop:
							break

					offs = next_offs

				else:
					# "Note", which finishes a row
					offs = next_offs

					if self._new_order is not None or self._new_row is not None:
						self._stop = True

					if self._stop:
						self._row_idx += 1

						break

					else:
						if self._def_dur is not None:
							self._row_idx += self._def_dur

						else:
							self._row_idx += mod._data[offs] + 1

							offs += 1
				
			pat_info = _PatternInfo(
				min(self._row_idx, song.info.pattern_length),
				self._tempo_changes,
				self._new_order,
				self._new_row,
			)
			pat_infos[pat_addr] = pat_info

		return

	def _calc_length(self, song: Song, pat_infos: Dict[int, _PatternInfo]) -> Tuple[float, float]:
		# "Play" it to time it
		speed = song.info.speed
		tempo = song.info.tempo
		order_idx = 0
		row_idx = 0
		total_frames = 0
		order_times = {}

		while order_idx not in order_times and row_idx >= 0:
			pat_addrs = song.frames[order_idx]

			# Get the pattern length and how it ended
			order_rows = 257
			next_order = next_row = None
			for chan_idx, pat_addr in enumerate(pat_addrs):
				pat_info = pat_infos[pat_addr]

				if pat_info.num_rows <= order_rows:
					order_rows = pat_info.num_rows

					if pat_info.next_order is not None or pat_info.next_row is not None:
						next_order = pat_info.next_order
						next_row = pat_info.next_row

			# Determine tempo changes
			frame_tempo_changes = sorted(itertools.chain.from_iterable(
					(pat_infos[addr].tempo_changes for addr in pat_addrs), 
				),
				key = lambda entry: entry.row_idx,
			)
			
			# Compute the number of frames for the pattern
			prev_frames = total_frames
			for entry in frame_tempo_changes:
				if entry.row_idx >= order_rows:
					break # Can happen if the composer does weird things

				num_rows = entry.row_idx - row_idx
				if num_rows:
					total_frames += num_rows * speed * 150 / tempo

					row_idx = entry.row_idx

				if entry.is_tempo:
					tempo = entry.value
				else:
					speed = entry.value

			total_frames += (order_rows - row_idx) * speed * 150 / tempo

			order_times[order_idx] = (prev_frames, total_frames)

			order_idx = next_order if next_order is not None else (order_idx + 1) % len(song.frames)
			row_idx = next_row or 0

		if row_idx >= 0:
			return (order_times[order_idx][0], total_frames - order_times[order_idx][0])
		else:
			return (total_frames, 0)

	_op_hdlrs = {0x82 + idx: hdlr for idx, hdlr in enumerate((
		_hdl_set_dur,
		_hdl_reset_dur,
		_hdl_speed,
		_hdl_tempo,
		_hdl_jump,
		_hdl_skip,
		_hdl_halt,
	))}
