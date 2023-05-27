.PHONY: all clean dir

SRCDIR := $(CURDIR)
STEM := $(notdir $(CURDIR))
BHOPDIR := bhop
BUILDDIR := build
CFG_NAME := $(STEM).cfg
ROM_NAME := $(STEM).nsf
DBG_NAME := $(STEM).dbg
LABELS_NAME := $(BUILDDIR)/labels.txt
MAP_NAME := $(BUILDDIR)/map.txt

# Assembler files, for building out the banks
ROOT_ASM_FILES := $(wildcard $(SRCDIR)/*.s)
BHOP_ASM_FILES := $(BHOPDIR)/bhop.s
O_FILES := \
  $(patsubst $(SRCDIR)/%.s,$(BUILDDIR)/%.o,$(ROOT_ASM_FILES)) \
  $(patsubst $(BHOPDIR)/%.s,$(BUILDDIR)/%.o,$(BHOP_ASM_FILES))

all: dir $(ROM_NAME)

dir:
	-@mkdir $(BUILDDIR)

clean:
	-@rm -rf build
	-@rm -f $(ROM_NAME)
	-@rm -f $(DBG_NAME)

$(ROM_NAME): $(CFG_NAME) $(O_FILES)
	ld65 -vm -m $(BUILDDIR)/map.txt -Ln $(BUILDDIR)/labels.txt --dbgfile $(DBG_NAME) -o $@ -C $^

$(BUILDDIR)/%.o: $(SRCDIR)/%.s
	ca65 -g -I $(BHOPDIR)/bhop -l $@.lst -o $@ $<

$(BUILDDIR)/%.o: $(BHOPDIR)/%.s
	ca65 -g -l $@.lst -o $@ $<
