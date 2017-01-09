Introduction
============

Context
=======

BOAST
-----

- General presentation of BOAST
- Purpose, capabilities, usage

MAQAO
-----

- General presentation of MAQAO
- Purpose, capabilities, usage

Integration
===========

- Big Picture
- Usage

Implementation
==============

Data Exchange
-------------

- Requirements
- YAML structured data
- BOAST --> MAQAO
- MAQAO --> BOAST

Kernel generation
-----------------

- Requirements
  - Executable kernel
  - No position-independent binary
  - Dwarf debug info
- Kernel wrapper

Kernel instrumentation
----------------------

- Principle
  - Memory access asm instruction rewriting
- Requirements
  - System V ABI + ARM ABI compliance
- ASM patching implementation
- Lua high-level instrumentation implementation
- MAQAO tracing library

Kernel analysis framework
-------------------------

- Principle
- C low-level framework
- Lua high-level framework
- SIMD analyzer example

Testcases examples
==================

- Basic vector addition kernel
- Mont-Blanc 2 Application BigDFT kernel

Conclusion and Future Work
==========================

