"""The ``memory_manipulator`` module is a wrapper around ``pymem`` for memory read and write access.

It implements some of the basic CheatEngine functionalities in Python. The game is controlled by
changing the values of ingame properties in the process memory. We cannot write to static memory
addresses since the process memory layout is dynamic and changes every time the game loads. Memory
locations are given as chains of pointers instead which we have to resolve to get the current
address for each attribute. These pointer chains were largely copied from available Dark Souls III
cheat tables.

Note:
    Not all game properties of interest were included in the cheat tables. Some values and their
    pointer chains were determined by us and are by no means guaranteed to be stable. Please report
    any memory read or write error to help us identify unstable pointer chains!

Warning:
    We cache resolved pointer chains to increase read and write access times. This requires manual
    cache clearing. For details see :meth:`MemoryManipulator.clear_cache`.

The ``MemoryManipulator`` is writing from an external process to a memory region in use by the game
process. You *will* see race conditions during writing, particularly for values with high frequency
writes in the game loop (e.g. coordinates). Be sure to include checks if writes were successful and
have taken effect in the game when you write to these memory locations.
"""
from __future__ import annotations

import platform
from typing import TypedDict, NotRequired

if platform.system() == "Windows":  # Windows imports, ignore for unix to make imports work
    import win32process
    import win32api
    import win32con

import pymem as pym
from pymem import Pymem

from soulsgym.core.utils import Singleton, get_pid
from soulsgym.core.static import address_base_patterns, address_bases


class AddressRecord(TypedDict):
    """Type definition for an address record."""
    base: str
    offsets: list[int]
    type: str
    length: NotRequired[int]  # Length of string or byte arrays
    codec: NotRequired[str]  # Codec for string decoding


class MemoryManipulator(metaclass=Singleton):
    """Handle reads and writes to the game process memory.

    The ``MemoryManipulator`` wraps ``pymem`` functions for memory read and writes. It manages the
    game memory pointers, address resolving and decoding.
    """

    def __init__(self, process_name: str = "DarkSoulsIII.exe"):
        """Initialize the cache and pointer attributes.

        If the game is not open, the pointer values can't be inferred which causes an exception.

        Args:
            process_name: The target process name. Should always be DarkSoulsIII.exe, unless the app
                name changes.
        """
        if not hasattr(self, "is_init"):
            self.process_name = process_name
            self.pid = get_pid(self.process_name)
            # Get the base address
            process_handle = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, False, self.pid)
            self.base_address = win32process.EnumProcessModules(process_handle)[0]
            # Create Pymem object once, this has a relative long initialziation
            self.pymem = Pymem()
            self.pymem.open_process_from_id(self.pid)
            self.address_cache: dict[str, int] = {}
            # Find the base addresses. Use static addresses where nothing else available. Else use
            # pymems AOB scan functions
            self.process_module = pym.process.module_from_name(self.pymem.process_handle,
                                                               self.process_name)
            self.bases = self._load_bases(process_name)

    def resolve_record(self, record: AddressRecord) -> int:
        """Resolve an address record by following its pointer chain to the final address.

        Resolved addresses are cached to increase performance. If the program reallocates memory,
        the cached addresses are no longer valid and the cache has to be cleared.

        Warning:
            Can't detect an invalid cache, this is the user's responsibility!

        Args:
            record: The address record. Address records must contain at least the `base` and
                `offsets` keys.

        Returns:
            The resolved address.
        """
        unique_address_id = str((record["offsets"], record["base"]))
        if unique_address_id in self.address_cache:  # Look up the cache first
            return self.address_cache[unique_address_id]
        # When no cache hit: resolve by following the pointer chain until its last link
        address = self.pymem.read_longlong(self.bases[record["base"]])
        for offset in record["offsets"][:-1]:
            address = self.pymem.read_longlong(address + offset)
        address += record["offsets"][-1]
        self.address_cache[unique_address_id] = address  # Add resolved address to cache
        return address

    def clear_cache(self):
        """Clear the reference look-up cache of the memory manipulator.

        The ``MemoryManipulator`` caches all pointer chains it resolves to speed up the reads and
        writes. If the game reloads, these addresses are no longer guaranteed to be valid and the
        address cache has to be cleared in order to resolve the new addresses of all values. Cache
        validation by reading the player death count is omitted since it incurs additional overhead
        for read operations and offsets any performance gains made by using an address cache.

        Warning:
            We do not validate the cache before reading from a cached address! It is the users's
            responsibility to clear the cache on reload!
        """
        self.address_cache = {}

    def read_record(self, record: AddressRecord) -> int | float | str | bytes:
        """Resolve the record address and read the value into the hinted type.

        Args:
            record: The address record.

        Returns:
            The read value.
        """
        address = self.resolve_record(record)
        match record["type"]:
            case "int":
                return self.read_int(address)
            case "float":
                return self.read_float(address)
            case "str":
                return self.read_string(address, length=record["length"], codec=record["codec"])
            case "bytes":
                return self.read_bytes(address, record["length"])
            case _:
                raise ValueError(f"Type '{record['type']}' not supported!")

    def write_record(self, record: AddressRecord, value: int | float | bytes):
        """Resolve the record address and write the value to the address.

        The provided value has to match the type hint of the record.

        Note:
            We support reading string records, but not writing them.

        Args:
            record: The address record.
            value: The value to write. Type has to match the type hint of the record.
        """
        address = self.resolve_record(record)
        match record["type"]:
            case "int":
                assert isinstance(value, int), f"Trying to write {type(value)} to int record!"
                self.write_int(address, value)
            case "float":
                assert isinstance(value, float) or isinstance(
                    value, int), f"Trying to write {type(value)} to float record!"
                self.write_float(address, value)
            case "bytes":
                assert isinstance(value, bytes), f"Trying to write {type(value)} to byte record!"
                self.write_bytes(address, value)
            case _:
                raise ValueError(f"Type '{record['type']}' not supported!")

    def read_int(self, address: int) -> int:
        """Read an integer from memory.

        Args:
            address: The read address.

        Returns:
            The integer value.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_long(address)

    def read_float(self, address: int) -> float:
        """Read a float from memory.

        Args:
            address: The read address.

        Returns:
            The float value.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_float(address)

    def read_string(self,
                    address: int,
                    length: int,
                    null_term: bool = True,
                    codec: str = "utf-16") -> str:
        """Read a string from memory.

        Args:
            address: The read address.
            length: The expected (maximum) string length.
            null_term: String should be cut after double 0x00.
            codec: The codec used to decode the bytes.

        Returns:
            The string.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
            UnicodeDecodeError: An error with the decoding of the read bytes occured.
        """
        s = self.pymem.read_bytes(address, length)
        if null_term:
            pos = 0
            for i in range(1, length, 2):
                if s[i - 1] == 0x00 and s[i] == 0x00:
                    pos = i
                    break
            s = s[:pos - 1]
            if not pos:
                s = s + bytes(1)  # Add null termination for strings which exceed 20 chars.
        return s.decode(codec)

    def read_bytes(self, address: int, length: int) -> bytes:
        """Read raw bytes from memory.

        Args:
            address: The read address.
            length: The bytes length.

        Returns:
            The raw bytes.

        Raises:
            pym.exception.MemoryReadError: An error with the memory read occured.
        """
        return self.pymem.read_bytes(address, length)

    def write_bit(self, address: int, index: int, value: int):
        """Write a single bit.

        Args:
            address: The write address.
            index: The index of the bit (0 ... 7).
            value: The value of the bit (0/1).

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        byte = self.read_bytes(address, 1)
        mask = (1 << index).to_bytes(1, "little")
        byte = (byte[0] & ~mask[0]).to_bytes(1, "little")
        if value:
            byte = (byte[0] | mask[0]).to_bytes(1, "little")
        self.write_bytes(address, byte)

    def write_int(self, address: int, value: int):
        """Write an integer to memory.

        Args:
            address: The write address.
            value: The value of the integer.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_long(self.pymem.process_handle, address, value)

    def write_float(self, address: int, value: float):
        """Write a float to memory.

        Args:
            address: The write address.
            value: The value of the float.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_float(self.pymem.process_handle, address, value)

    def write_bytes(self, address: int, buffer: bytes):
        """Write a series of bytes to memory.

        Args:
            address: The write address for the first byte.
            buffer: The bytes.

        Raises:
            pym.exception.MemoryWriteError: An error with the memory write occured.
        """
        pym.memory.write_bytes(self.pymem.process_handle, address, buffer, len(buffer))

    def _load_bases(self, process_name: str) -> dict:
        match process_name:
            case "DarkSoulsIII.exe":
                game = "DarkSoulsIII"
            case "eldenring.exe":  # Not an error, eldenring.exe isn't capitalized
                game = "EldenRing"
            case "TEKKEN 8.exe":
                game = "Tekken8"
                return {}
            case _:
                raise ValueError(f"Process name '{process_name}' not supported!")
        if address_bases[game] is None:
            bases = {}
        else:
            bases = {name: addr + self.base_address for name, addr in address_bases[game].items()}
        for base_key, base in address_base_patterns[game].items():
            pattern = bytes(base["pattern"], "ASCII")
            addr = pym.pattern.pattern_scan_module(self.pymem.process_handle, self.process_module,
                                                   pattern)
            if not addr:
                raise RuntimeError(f"Pattern for '{base_key}' could not be resolved!")
            if "offset" in base:
                addr += base["offset"]
            # Conversion logic from TGA cheat table for Dark Souls III v. 3.1.2
            # More recent table versions use CE disassembler. Address is read from asm, e.g.
            #
            # rbx,[address]
            #
            # [x + 3] + 7 is necessary to extract the same addresses as CheatEngine. The + 3 should
            # be the offset for the leading asm command, e.g. rbx. Not sure about the +7, but it
            # matches the definition in version 3.1.2 and earlier.
            #
            # After the conversion, the address is equal to the base address in CheatEngine. I.e. if
            # the pointer chain is [["DarkSoulsIII.exe" + 0x123] + 0x456], the address we store in
            # bases is "DarkSoulsIII.exe" + 0x123.
            # TODO: If possible, replace with own disassembler
            bases[base_key] = addr + self.pymem.read_long(addr + 3) + 7
        return bases
