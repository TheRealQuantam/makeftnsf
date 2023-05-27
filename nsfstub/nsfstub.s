.include "bhop.inc"

.global __BANK0_1_LOAD__, bhop_music_data

bhop_music_data := $a000

.segment "NSF_HDR"
	.byte "NESM", $1a
	.byte 1 ; Version
	.byte 0 ; Number of songs
	.byte 1 ; Initial song
	.addr __BANK0_1_LOAD__ ; Padding after header until the first bank starts
	.addr Init ; Init address
	.addr Play ; Play address
	.res $20, 0 ; Song name
	.res $20, 0 ; Artist name
	.res $20, 0 ; Copyright name
	.word 16639 ; NTSC play speed. Actual NTSC NES rate.
	.byte 0, 1, 0, 0, 0, 0, 0, 0 ; Initial banks
	.word 0 ; PAL play speed
	.byte 0 ; PAL/NTSC flags. NTSC only.
	.byte 0 ; Expansion chips. No expansion chips.
	.byte 0 ; Reserved for NSF2
	.res 3, 0 ; Program data size (excluding metadata)
	
.segment "ZEROPAGE" : zeropage
Temp0: .byte 0
Temp1: .byte 0

;IsPlaying: .byte 0
CurBank: .byte 0

.segment "BANK0_1"
TrackMapTable:
	.res $200, 0

.proc PlayStub
	jmp bhop_play
.endproc

.proc Init 
	; A: 0-based song number
	
	; Stop playback before changing the banks in case the NMI causes Play to be called.
	ldx #$0
	;stx IsPlaying
	stx Temp1
	
	; Stop playback of anything previously playing
	;stx ApuStatus_4015
	
	;ldx #$f
	;stx ApuStatus_4015
	
	; Load the track map entry
	asl A
	rol Temp1
	
	clc
	adc #<TrackMapTable
	sta Temp0
	
	lda Temp1
	adc #>TrackMapTable
	sta Temp1
	
	; Switch the banks
	ldy #$0
	lda (Temp0), Y
	sta CurBank
	
	tax
	stx $5ffa
	inx
	stx $5ffb
	
	; Play the desired track
	iny
	lda (Temp0), Y
	
	jsr bhop_init
	
	;lda #$ff
	;sta IsPlaying
	
	rts
.endproc ; Init

.proc Play
	;lda IsPlaying
	;beq @Done
	
	jsr PlayStub
	
;@Done:
	rts
.endproc ; Play

