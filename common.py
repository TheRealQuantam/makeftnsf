# common.py
# Copyright 2024 Justin Olbrantz (Quantam)

# Common utilities.

# This Source Code Form is subject to the terms of the Mozilla Public License, v. 2.0. If a copy of the MPL was not distributed with this file, You can obtain one at https://mozilla.org/MPL/2.0/.

from __future__ import annotations
import collections as colls
import ctypes
from ctypes import c_char, c_byte, c_int8, c_uint8, c_int16, c_uint16, Structure, BigEndianStructure, LittleEndianStructure, sizeof, addressof
from enum import Enum, IntEnum, IntFlag, auto
import itertools
import logging
from pathlib import Path
from pydantic.dataclasses import dataclass
from pydantic import BaseModel as PydanticBaseModel, conint, conlist, constr, Field, parse_obj_as, PositiveInt, root_validator, StrictBool, StrictBytes, StrictFloat, StrictInt, StrictStr, validator
from typing import *
from typing_extensions import *

c_uint16_be = c_uint16.__ctype_be__
c_uint16_le = c_uint16.__ctype_le__

ines_hdr_sig = b"NES\x1a"
ines2_sig = 2

class iNesBasicHeader(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("sig", c_char * 4),
		("num_prg_16kbs", c_uint8),
		("num_chr_8kbs", c_uint8),
		
		("vertical_mirroring", c_uint8, 1),
		("has_battery", c_uint8, 1),
		("has_trainer", c_uint8, 1),
		("four_screen_vram", c_uint8, 1),
		("mapper_lo_nibble", c_uint8, 4),

		("vs_unisystem", c_uint8, 1),
		("play_choice_10", c_uint8, 1),
		("ines2_sig", c_uint8, 2),
		("mapper_mid_nibble", c_uint8, 4),
	)

class iNes20Header(iNesBasicHeader):
	_fields_ = (
		("mapper_hi_nibble", c_uint8, 4),
		("submapper", c_uint8, 4),

		("num_prg_16kbs_hi", c_uint8, 4),
		("num_chr_8kbs_hi", c_uint8, 4),

		("prg_ram_shift", c_uint8, 4),
		("prg_nvram_shift", c_uint8, 4),

		("chr_ram_shift", c_uint8, 4),
		("chr_nvram_shift", c_uint8, 4),

		("flags_c", c_uint8),
		("flags_d", c_uint8),
		("flags_e", c_uint8),
		("flags_f", c_uint8),
	)

class iNesDecodedHeader(NamedTuple):
	mapper: int
	submapper: int
	prg_offset: int
	num_prg_16kbs: int
	chr_offset: int
	num_chr_8kbs: int
	prg_ram_size: int
	prg_nvram_size: int
	chr_ram_size: int
	chr_nvram_size: int
	vertical_mirroring: bool
	has_battery: bool
	has_trainer: bool
	four_screen_vram: bool
	vs_unisystem: bool
	play_choice_10: bool

class BankOffset(NamedTuple):
	bank_idx: int
	offset: int

class Range(NamedTuple):
	start: int
	end: Optional[int]

class UniqueRange(NamedTuple):
	bank_idx: int
	base_addr: int
	start: int
	end: Optional[int]

class MasterTrackTableEntry(LittleEndianStructure):
	_pack_ = True
	_fields_ = (
		("bank_idx", c_uint8),
		("track_idx", c_uint8),
	)

class FreeSpaceEntry(NamedTuple):
	bank_idx: int
	base_addr: int
	offset: int
	size: int

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
	) -> DataError:
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

class FatalError(Exception):
	pass

class WrappedFatalError(FatalError):
	def __init__(self, err):
		super().__init__(err)

	def __str__(self):
		return str(self.args[0])

	def __repr__(self):
		return repr(self.args[0])

def get_leca(base_addr: int, bank_size: int) -> Callable[[int, int], int]:
	def leca(bank, addr):
		offs = addr - base_addr
		return bank * bank_size + offs + 0x10

	return leca

def check_addr(
	addr: int, 
	base_addr: int, 
	size_or_data: Union[int, ByteString],
) -> bool:
	if isinstance(size_or_data, int):
		size = size_or_data
	else:
		size = len(size_or_data)

	return addr >= base_addr and addr <= base_addr + size

def decode_ines_header(
	rom: ByteString, 
	rom_size: Optional[int] = None,
) -> iNesDecodedHeader:
	if rom_size is None:
		rom_size = len(rom)

	hdr = iNes20Header.from_buffer_copy(rom)
	if hdr.sig != ines_hdr_sig:
		raise FormatError.from_field("Not an iNES ROM", rom, hdr, "sig")

	map_idx = hdr.mapper_lo_nibble
	submap_idx = 0
	num_prg_16kbs = hdr.num_prg_16kbs
	num_chr_8kbs = hdr.num_chr_8kbs

	prg_offset = sizeof(iNes20Header)
	if hdr.has_trainer:
		prg_offset += 512

	num_prg_16kbs_2 = num_prg_16kbs + hdr.num_prg_16kbs_hi * 0x100
	num_chr_8kbs_2 = num_chr_8kbs + hdr.num_chr_8kbs_hi * 0x100
	calc_size = prg_offset + (num_prg_16kbs_2 * 0x4000) + (num_chr_8kbs_2 * 0x2000)
	if hdr.ines2_sig == ines2_sig and calc_size <= rom_size:
		version = 0x20
	elif hdr.ines2_sig == 1:
		version = 0
	elif hdr.ines2_sig == 0 and not any(rom[12:16]):
		version = 0x10
	else:
		version = 0

	prg_ram_size, prg_nvram_size = (0, 0x2000) if hdr.has_battery else (0x2000, 0)
	chr_ram_size = 0x2000 if num_chr_8kbs == 0 else 0
	chr_nvram_size = 0
	if version >= 0x10:
		map_idx += hdr.mapper_mid_nibble * 0x10
	
	if version >= 0x20:
		map_idx += hdr.mapper_hi_nibble * 0x100
		submap_idx = hdr.submapper

		num_prg_16kbs = num_prg_16kbs_2
		num_chr_8kbs = num_chr_8kbs_2

		prg_ram_size = 64 << hdr.prg_ram_shift
		prg_nvram_size = 64 << hdr.prg_nvram_shift
		chr_ram_size = 64 << hdr.chr_ram_shift
		chr_nvram_size = 64 << hdr.chr_nvram_shift

	chr_offset = prg_offset + num_prg_16kbs * 0x4000
	if chr_offset > rom_size:
		raise FormatError.from_field("PRG-ROM size exceeds size of ROM", rom, hdr, "num_prg_16kbs")

	if chr_offset + num_chr_8kbs * 0x2000 > rom_size:
		raise FormatError.from_field("CHR-ROM size exceeds size of ROM", rom, hdr, "num_chr_8kbs")

	return iNesDecodedHeader(
		map_idx,
		submap_idx,
		prg_offset,
		num_prg_16kbs,
		chr_offset,
		num_chr_8kbs,
		prg_ram_size,
		prg_nvram_size,
		chr_ram_size,
		chr_nvram_size,
		bool(hdr.vertical_mirroring),
		bool(hdr.has_battery),
		bool(hdr.has_trainer),
		bool(hdr.four_screen_vram),
		bool(hdr.vs_unisystem),
		bool(hdr.play_choice_10),
	)

class TrackTypes(Enum):
	Excluded = auto()
	Unused = auto()
	BuiltIn = auto()
	Imported = auto()

class FileInfo:
	def __init__(
		self, 
		fmt: str, 
		*, 
		key: Hashable = None,
		base_addr: Optional[int] = None,
		path: Optional[Union[Path, str]] = None,
		data: Optional[Union[ByteString, str]] = None,
		size: Optional[int] = None,
		res_size: Optional[int] = None,
		base_path: Optional[Path] = None,
		) -> None:
		self.format = fmt
		self.tracks = []
		self.base_addr = base_addr # The base address of the data

		if isinstance(path, str):
			path = base_path.joinpath(path) if base_path else Path(path)

		self.path = self.orig_path = path
		self.data = bytes.fromhex(data) if isinstance(data, str) else data
		self.size = size if size is not None else (len(self.data) if data else None)
		self.res_size = res_size if res_size is not None else self.size

		self.bank_idx = None
		self.address = None # The address in bank_idx it's actually imported at

		self.key = key if key is not None else path or id(self)

class TrackInfo:
	def __init__(
		self, 
		name: str, 
		file: Optional[FileInfo], 
		index: int, 
		opts: Optional[dict] = None, 
		*, 
		ty: TrackTypes = TrackTypes.Imported, 
		is_sfx: bool = False,
	) -> None:
		self.name = name
		self.file = file
		self.index = index

		self.type = ty
		self.is_sfx = is_sfx

		self.opts = opts or {}
