D2_LAYOUT := elk
D2_SRCS := $(wildcard docs/*.d2)
D2_OUTS := $(D2_SRCS:.d2=.svg)

.PHONY: diagrams
diagrams: $(D2_OUTS)

docs/%.svg: docs/%.d2
	d2 --layout $(D2_LAYOUT) $< $@
