makeftnsf

By Justin Olbrantz (Quantam)

makeftnsf is a tool for the Mega Man 2 Randomizer that converts the list of FTMs in FtSoundTrackConfiguration.xml into a working NSF that plays the FTMs through bhop. This allows those adding tracks to the randomizer to preview the tracks to ensure they play correctly in bhop - which does not support the full FT feature set - as well as to measure the CPU consumed by each track; CPU usage is usually not a significant issue, but it may be a problem for FTMs that were auto-converted from NSFs.

However, apart from the specific bhop version used, lack of DPCM support, and input format, no portion of makeftnsf is specific to mm2ft, and it may be used to preview tracks for other uses of bhop unrelated to Mega Man or MM2R.

HOW TO USE IT

The easiest way to use makeftnsf is simply to put the FtSoundTrackConfiguration.xml file into the same directory as the Python files and run makeftnsf.py. This will generate FtSoundTrackConfiguration.nsf and FtSoundTrackConfiguration.nsfe. NSFe files include track titles and authors, though they are less well-supported (e.g. on flash carts) than NSF files.

LINKS

Mega Man 2 Randomizer repo: https://github.com/squid-man/MegaMan2Randomizer2
Primary makeftnsf repo: https://github.com/TheRealQuantam/makeftnsf
bhop repo: https://github.com/zeta0134/bhop