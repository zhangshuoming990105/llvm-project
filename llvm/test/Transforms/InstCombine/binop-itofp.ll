; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -passes=instcombine -S | FileCheck %s

define half @test_ui_ui_i8_add(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i8_add(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 127
; CHECK-NEXT:    [[Y:%.*]] = and i8 [[Y_IN:%.*]], 127
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 127
  %y = and i8 %y_in, 127
  %xf = uitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i8_add_fail_overflow(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i8_add_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 127
; CHECK-NEXT:    [[Y:%.*]] = and i8 [[Y_IN:%.*]], -127
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = uitofp i8 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 127
  %y = and i8 %y_in, 129
  %xf = uitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i8_add_C(i8 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i8_add_C(
; CHECK-NEXT:    [[TMP1:%.*]] = or i8 [[X_IN:%.*]], -128
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 127
  %xf = uitofp i8 %x to half
  %r = fadd half %xf, 128.0
  ret half %r
}

define half @test_ui_ui_i8_add_C_fail_no_repr(i8 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i8_add_C_fail_no_repr(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 127
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], 0xH57F8
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 127
  %xf = uitofp i8 %x to half
  %r = fadd half %xf, 127.5
  ret half %r
}

define half @test_ui_ui_i8_add_C_fail_overflow(i8 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i8_add_C_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 127
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], 0xH5808
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 127
  %xf = uitofp i8 %x to half
  %r = fadd half %xf, 129.0
  ret half %r
}

define half @test_si_si_i8_add(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i8_add(
; CHECK-NEXT:    [[X:%.*]] = or i8 [[X_IN:%.*]], -64
; CHECK-NEXT:    [[Y:%.*]] = or i8 [[Y_IN:%.*]], -64
; CHECK-NEXT:    [[TMP1:%.*]] = add nsw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i8 %x_in, -64
  %y = or i8 %y_in, -64
  %xf = sitofp i8 %x to half
  %yf = sitofp i8 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_si_si_i8_add_fail_overflow(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i8_add_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = or i8 [[X_IN:%.*]], -64
; CHECK-NEXT:    [[Y:%.*]] = or i8 [[Y_IN:%.*]], -65
; CHECK-NEXT:    [[XF:%.*]] = sitofp i8 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i8 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i8 %x_in, -64
  %y = or i8 %y_in, -65
  %xf = sitofp i8 %x to half
  %yf = sitofp i8 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_ui_si_i8_add(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i8_add(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 63
; CHECK-NEXT:    [[Y:%.*]] = and i8 [[Y_IN:%.*]], 63
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw nsw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 63
  %y = and i8 %y_in, 63
  %xf = sitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_ui_si_i8_add_overflow(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i8_add_overflow(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 63
; CHECK-NEXT:    [[Y:%.*]] = and i8 [[Y_IN:%.*]], 65
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 63
  %y = and i8 %y_in, 65
  %xf = sitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i8_sub_C(i8 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i8_sub_C(
; CHECK-NEXT:    [[TMP1:%.*]] = and i8 [[X_IN:%.*]], 127
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i8 %x_in, 128
  %xf = uitofp i8 %x to half
  %r = fsub half %xf, 128.0
  ret half %r
}

define half @test_ui_ui_i8_sub_C_fail_overflow(i8 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i8_sub_C_fail_overflow(
; CHECK-NEXT:    [[TMP1:%.*]] = or i8 [[X_IN:%.*]], -128
; CHECK-NEXT:    [[R:%.*]] = sitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 127
  %xf = uitofp i8 %x to half
  %r = fsub half %xf, 128.0
  ret half %r
}

define half @test_si_si_i8_sub(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i8_sub(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 63
; CHECK-NEXT:    [[Y:%.*]] = or i8 [[Y_IN:%.*]], -64
; CHECK-NEXT:    [[TMP1:%.*]] = sub nsw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 63
  %y = or i8 %y_in, -64
  %xf = sitofp i8 %x to half
  %yf = sitofp i8 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_si_si_i8_sub_fail_overflow(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i8_sub_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 63
; CHECK-NEXT:    [[Y:%.*]] = or i8 [[Y_IN:%.*]], -65
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i8 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fsub half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 63
  %y = or i8 %y_in, -65
  %xf = sitofp i8 %x to half
  %yf = sitofp i8 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_si_si_i8_sub_C(i8 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i8_sub_C(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 63
; CHECK-NEXT:    [[TMP1:%.*]] = or disjoint i8 [[X]], 64
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 63
  %xf = sitofp i8 %x to half
  %r = fsub half %xf, -64.0
  ret half %r
}

define half @test_si_si_i8_sub_C_fail_overflow(i8 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i8_sub_C_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 65
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw i8 [[X]], 64
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 65
  %xf = sitofp i8 %x to half
  %r = fsub half %xf, -64.0
  ret half %r
}

define half @test_ui_si_i8_sub(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i8_sub(
; CHECK-NEXT:    [[X:%.*]] = or i8 [[X_IN:%.*]], 64
; CHECK-NEXT:    [[Y:%.*]] = and i8 [[Y_IN:%.*]], 63
; CHECK-NEXT:    [[TMP1:%.*]] = sub nuw nsw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i8 %x_in, 64
  %y = and i8 %y_in, 63
  %xf = sitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_ui_si_i8_sub_fail_maybe_sign(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i8_sub_fail_maybe_sign(
; CHECK-NEXT:    [[X:%.*]] = or i8 [[X_IN:%.*]], 64
; CHECK-NEXT:    [[Y:%.*]] = and i8 [[Y_IN:%.*]], 63
; CHECK-NEXT:    [[TMP1:%.*]] = sub nuw nsw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i8 %x_in, 64
  %y = and i8 %y_in, 63
  %xf = uitofp i8 %x to half
  %yf = sitofp i8 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i8_mul(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i8_mul(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 15
; CHECK-NEXT:    [[Y:%.*]] = and i8 [[Y_IN:%.*]], 15
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 15
  %y = and i8 %y_in, 15
  %xf = uitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i8_mul_C(i8 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i8_mul_C(
; CHECK-NEXT:    [[X:%.*]] = shl i8 [[X_IN:%.*]], 4
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 15
  %xf = uitofp i8 %x to half
  %r = fmul half %xf, 16.0
  ret half %r
}

define half @test_ui_ui_i8_mul_C_fail_overlow(i8 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i8_mul_C_fail_overlow(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 14
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], 0xH4CC0
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 14
  %xf = uitofp i8 %x to half
  %r = fmul half %xf, 19.0
  ret half %r
}

define half @test_si_si_i8_mul(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i8_mul(
; CHECK-NEXT:    [[XX:%.*]] = and i8 [[X_IN:%.*]], 6
; CHECK-NEXT:    [[X:%.*]] = or disjoint i8 [[XX]], 1
; CHECK-NEXT:    [[Y:%.*]] = or i8 [[Y_IN:%.*]], -8
; CHECK-NEXT:    [[TMP1:%.*]] = mul nsw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i8 %x_in, 6
  %x = add nsw nuw i8 %xx, 1
  %y = or i8 %y_in, -8
  %xf = sitofp i8 %x to half
  %yf = sitofp i8 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_si_si_i8_mul_fail_maybe_zero(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i8_mul_fail_maybe_zero(
; CHECK-NEXT:    [[X:%.*]] = and i8 [[X_IN:%.*]], 7
; CHECK-NEXT:    [[Y:%.*]] = or i8 [[Y_IN:%.*]], -8
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i8 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i8 %x_in, 7
  %y = or i8 %y_in, -8
  %xf = sitofp i8 %x to half
  %yf = sitofp i8 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_si_si_i8_mul_C_fail_no_repr(i8 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i8_mul_C_fail_no_repr(
; CHECK-NEXT:    [[XX:%.*]] = and i8 [[X_IN:%.*]], 6
; CHECK-NEXT:    [[X:%.*]] = or disjoint i8 [[XX]], 1
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], 0xHC780
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i8 %x_in, 6
  %x = add nsw nuw i8 %xx, 1
  %xf = sitofp i8 %x to half
  %r = fmul half %xf, -7.5
  ret half %r
}

define half @test_si_si_i8_mul_C_fail_overflow(i8 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i8_mul_C_fail_overflow(
; CHECK-NEXT:    [[XX:%.*]] = and i8 [[X_IN:%.*]], 6
; CHECK-NEXT:    [[X:%.*]] = or disjoint i8 [[XX]], 1
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], 0xHCCC0
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i8 %x_in, 6
  %x = add nsw nuw i8 %xx, 1
  %xf = sitofp i8 %x to half
  %r = fmul half %xf, -19.0
  ret half %r
}

define half @test_ui_si_i8_mul(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i8_mul(
; CHECK-NEXT:    [[XX:%.*]] = and i8 [[X_IN:%.*]], 6
; CHECK-NEXT:    [[X:%.*]] = or disjoint i8 [[XX]], 1
; CHECK-NEXT:    [[YY:%.*]] = and i8 [[Y_IN:%.*]], 7
; CHECK-NEXT:    [[Y:%.*]] = add nuw nsw i8 [[YY]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw nsw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i8 %x_in, 6
  %x = add i8 %xx, 1
  %yy = and i8 %y_in, 7
  %y = add i8 %yy, 1
  %xf = sitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_ui_si_i8_mul_fail_maybe_zero(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i8_mul_fail_maybe_zero(
; CHECK-NEXT:    [[XX:%.*]] = and i8 [[X_IN:%.*]], 7
; CHECK-NEXT:    [[X:%.*]] = add nuw nsw i8 [[XX]], 1
; CHECK-NEXT:    [[Y:%.*]] = and i8 [[Y_IN:%.*]], 7
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw nsw i8 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i8 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i8 %x_in, 7
  %x = add i8 %xx, 1
  %y = and i8 %y_in, 7
  %xf = sitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_ui_si_i8_mul_fail_signed(i8 noundef %x_in, i8 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i8_mul_fail_signed(
; CHECK-NEXT:    [[XX:%.*]] = and i8 [[X_IN:%.*]], 7
; CHECK-NEXT:    [[X:%.*]] = add nuw nsw i8 [[XX]], 1
; CHECK-NEXT:    [[Y:%.*]] = or i8 [[Y_IN:%.*]], -4
; CHECK-NEXT:    [[XF:%.*]] = uitofp i8 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = uitofp i8 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i8 %x_in, 7
  %x = add i8 %xx, 1
  %y = or i8 %y_in, -4
  %xf = sitofp i8 %x to half
  %yf = uitofp i8 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i16_add(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i16_add(
; CHECK-NEXT:    [[X:%.*]] = and i16 [[X_IN:%.*]], 2047
; CHECK-NEXT:    [[Y:%.*]] = and i16 [[Y_IN:%.*]], 2047
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw nsw i16 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i16 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i16 %x_in, 2047
  %y = and i16 %y_in, 2047
  %xf = uitofp i16 %x to half
  %yf = uitofp i16 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i16_add_fail_not_promotable(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i16_add_fail_not_promotable(
; CHECK-NEXT:    [[X:%.*]] = and i16 [[X_IN:%.*]], 2049
; CHECK-NEXT:    [[Y:%.*]] = and i16 [[Y_IN:%.*]], 2047
; CHECK-NEXT:    [[XF:%.*]] = uitofp i16 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = uitofp i16 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i16 %x_in, 2049
  %y = and i16 %y_in, 2047
  %xf = uitofp i16 %x to half
  %yf = uitofp i16 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i16_add_C(i16 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i16_add_C(
; CHECK-NEXT:    [[TMP1:%.*]] = or i16 [[X_IN:%.*]], -2048
; CHECK-NEXT:    [[R:%.*]] = uitofp i16 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i16 %x_in, 2047
  %xf = uitofp i16 %x to half
  %r = fadd half %xf, 63488.0
  ret half %r
}

define half @test_ui_ui_i16_add_C_fail_overflow(i16 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i16_add_C_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = and i16 [[X_IN:%.*]], 2047
; CHECK-NEXT:    [[XF:%.*]] = uitofp i16 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], 0xH7BD0
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i16 %x_in, 2047
  %xf = uitofp i16 %x to half
  %r = fadd half %xf, 64000.0
  ret half %r
}

define half @test_si_si_i16_add(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i16_add(
; CHECK-NEXT:    [[X:%.*]] = or i16 [[X_IN:%.*]], -2048
; CHECK-NEXT:    [[Y:%.*]] = or i16 [[Y_IN:%.*]], -2048
; CHECK-NEXT:    [[TMP1:%.*]] = add nsw i16 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i16 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i16 %x_in, -2048
  %y = or i16 %y_in, -2048
  %xf = sitofp i16 %x to half
  %yf = sitofp i16 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_si_si_i16_add_fail_no_promotion(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i16_add_fail_no_promotion(
; CHECK-NEXT:    [[XX:%.*]] = or i16 [[X_IN:%.*]], -2048
; CHECK-NEXT:    [[X:%.*]] = add nsw i16 [[XX]], -1
; CHECK-NEXT:    [[Y:%.*]] = or i16 [[Y_IN:%.*]], -2048
; CHECK-NEXT:    [[XF:%.*]] = sitofp i16 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i16 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %xx = or i16 %x_in, -2048
  %x = sub i16 %xx, 1
  %y = or i16 %y_in, -2048
  %xf = sitofp i16 %x to half
  %yf = sitofp i16 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_si_si_i16_add_C_overflow(i16 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i16_add_C_overflow(
; CHECK-NEXT:    [[X:%.*]] = or i16 [[X_IN:%.*]], -2048
; CHECK-NEXT:    [[XF:%.*]] = sitofp i16 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], 0xH7840
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i16 %x_in, -2048
  %xf = sitofp i16 %x to half
  %r = fadd half %xf, 0xH7840
  ret half %r
}

define half @test_si_si_i16_sub(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i16_sub(
; CHECK-NEXT:    [[X:%.*]] = or i16 [[X_IN:%.*]], -2048
; CHECK-NEXT:    [[Y:%.*]] = and i16 [[Y_IN:%.*]], 2047
; CHECK-NEXT:    [[TMP1:%.*]] = sub nuw nsw i16 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i16 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i16 %x_in, -2048
  %y = and i16 %y_in, 2047
  %xf = sitofp i16 %x to half
  %yf = sitofp i16 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_si_si_i16_sub_fail_no_promotion(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i16_sub_fail_no_promotion(
; CHECK-NEXT:    [[X:%.*]] = and i16 [[X_IN:%.*]], 2047
; CHECK-NEXT:    [[Y:%.*]] = or i16 [[Y_IN:%.*]], -2049
; CHECK-NEXT:    [[XF:%.*]] = uitofp i16 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i16 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fsub half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i16 %x_in, 2047
  %y = or i16 %y_in, -2049
  %xf = sitofp i16 %x to half
  %yf = sitofp i16 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_ui_si_i16_sub(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i16_sub(
; CHECK-NEXT:    [[X:%.*]] = and i16 [[X_IN:%.*]], 2047
; CHECK-NEXT:    [[Y:%.*]] = and i16 [[Y_IN:%.*]], 2047
; CHECK-NEXT:    [[TMP1:%.*]] = sub nsw i16 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i16 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i16 %x_in, 2047
  %y = and i16 %y_in, 2047
  %xf = uitofp i16 %x to half
  %yf = sitofp i16 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_ui_si_i16_sub_fail_maybe_signed(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i16_sub_fail_maybe_signed(
; CHECK-NEXT:    [[X:%.*]] = or i16 [[X_IN:%.*]], -2048
; CHECK-NEXT:    [[Y:%.*]] = and i16 [[Y_IN:%.*]], 2047
; CHECK-NEXT:    [[XF:%.*]] = uitofp i16 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = uitofp i16 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fsub half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i16 %x_in, -2048
  %y = and i16 %y_in, 2047
  %xf = uitofp i16 %x to half
  %yf = sitofp i16 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i16_mul(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i16_mul(
; CHECK-NEXT:    [[X:%.*]] = and i16 [[X_IN:%.*]], 255
; CHECK-NEXT:    [[Y:%.*]] = and i16 [[Y_IN:%.*]], 255
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw i16 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i16 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i16 %x_in, 255
  %y = and i16 %y_in, 255
  %xf = uitofp i16 %x to half
  %yf = uitofp i16 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i16_mul_fail_no_promotion(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i16_mul_fail_no_promotion(
; CHECK-NEXT:    [[X:%.*]] = and i16 [[X_IN:%.*]], 4095
; CHECK-NEXT:    [[Y:%.*]] = and i16 [[Y_IN:%.*]], 3
; CHECK-NEXT:    [[XF:%.*]] = uitofp i16 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = uitofp i16 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i16 %x_in, 4095
  %y = and i16 %y_in, 3
  %xf = uitofp i16 %x to half
  %yf = uitofp i16 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_si_si_i16_mul(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i16_mul(
; CHECK-NEXT:    [[XX:%.*]] = and i16 [[X_IN:%.*]], 126
; CHECK-NEXT:    [[X:%.*]] = or disjoint i16 [[XX]], 1
; CHECK-NEXT:    [[Y:%.*]] = or i16 [[Y_IN:%.*]], -255
; CHECK-NEXT:    [[TMP1:%.*]] = mul nsw i16 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i16 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i16 %x_in, 126
  %x = add nsw nuw i16 %xx, 1
  %y = or i16 %y_in, -255
  %xf = sitofp i16 %x to half
  %yf = sitofp i16 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_si_si_i16_mul_fail_overflow(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i16_mul_fail_overflow(
; CHECK-NEXT:    [[XX:%.*]] = and i16 [[X_IN:%.*]], 126
; CHECK-NEXT:    [[X:%.*]] = or disjoint i16 [[XX]], 1
; CHECK-NEXT:    [[Y:%.*]] = or i16 [[Y_IN:%.*]], -257
; CHECK-NEXT:    [[XF:%.*]] = uitofp i16 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i16 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i16 %x_in, 126
  %x = add nsw nuw i16 %xx, 1
  %y = or i16 %y_in, -257
  %xf = sitofp i16 %x to half
  %yf = sitofp i16 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_si_si_i16_mul_C_fail_overflow(i16 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i16_mul_C_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = or i16 [[X_IN:%.*]], -129
; CHECK-NEXT:    [[XF:%.*]] = sitofp i16 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], 0xH5800
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i16 %x_in, -129
  %xf = sitofp i16 %x to half
  %r = fmul half %xf, 128.0
  ret half %r
}

define half @test_si_si_i16_mul_C_fail_no_promotion(i16 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i16_mul_C_fail_no_promotion(
; CHECK-NEXT:    [[X:%.*]] = or i16 [[X_IN:%.*]], -4097
; CHECK-NEXT:    [[XF:%.*]] = sitofp i16 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], 0xH4500
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i16 %x_in, -4097
  %xf = sitofp i16 %x to half
  %r = fmul half %xf, 5.0
  ret half %r
}

define half @test_ui_si_i16_mul(i16 noundef %x_in, i16 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i16_mul(
; CHECK-NEXT:    [[XX:%.*]] = and i16 [[X_IN:%.*]], 126
; CHECK-NEXT:    [[X:%.*]] = or disjoint i16 [[XX]], 1
; CHECK-NEXT:    [[YY:%.*]] = and i16 [[Y_IN:%.*]], 126
; CHECK-NEXT:    [[Y:%.*]] = or disjoint i16 [[YY]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw nsw i16 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i16 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i16 %x_in, 126
  %x = add i16 %xx, 1
  %yy = and i16 %y_in, 126
  %y = add i16 %yy, 1
  %xf = sitofp i16 %x to half
  %yf = uitofp i16 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i12_add(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i12_add(
; CHECK-NEXT:    [[X:%.*]] = and i12 [[X_IN:%.*]], 2047
; CHECK-NEXT:    [[Y:%.*]] = and i12 [[Y_IN:%.*]], 2047
; CHECK-NEXT:    [[TMP1:%.*]] = add nuw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i12 %x_in, 2047
  %y = and i12 %y_in, 2047
  %xf = uitofp i12 %x to half
  %yf = uitofp i12 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i12_add_fail_overflow(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i12_add_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = and i12 [[X_IN:%.*]], 2047
; CHECK-NEXT:    [[Y:%.*]] = and i12 [[Y_IN:%.*]], -2047
; CHECK-NEXT:    [[XF:%.*]] = uitofp i12 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = uitofp i12 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i12 %x_in, 2047
  %y = and i12 %y_in, 2049
  %xf = uitofp i12 %x to half
  %yf = uitofp i12 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}


define half @test_si_si_i12_add(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i12_add(
; CHECK-NEXT:    [[X:%.*]] = or i12 [[X_IN:%.*]], -1024
; CHECK-NEXT:    [[Y:%.*]] = or i12 [[Y_IN:%.*]], -1024
; CHECK-NEXT:    [[TMP1:%.*]] = add nsw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i12 %x_in, -1024
  %y = or i12 %y_in, -1024
  %xf = sitofp i12 %x to half
  %yf = sitofp i12 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_si_si_i12_add_fail_overflow(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i12_add_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = or i12 [[X_IN:%.*]], -1025
; CHECK-NEXT:    [[Y:%.*]] = or i12 [[Y_IN:%.*]], -1025
; CHECK-NEXT:    [[XF:%.*]] = sitofp i12 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i12 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i12 %x_in, -1025
  %y = or i12 %y_in, -1025
  %xf = sitofp i12 %x to half
  %yf = sitofp i12 %y to half
  %r = fadd half %xf, %yf
  ret half %r
}

define half @test_si_si_i12_add_C_fail_overflow(i12 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i12_add_C_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = or i12 [[X_IN:%.*]], -2048
; CHECK-NEXT:    [[XF:%.*]] = sitofp i12 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fadd half [[XF]], 0xHBC00
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i12 %x_in, -2048
  %xf = sitofp i12 %x to half
  %r = fadd half %xf, -1.0
  ret half %r
}

define half @test_ui_ui_i12_sub(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i12_sub(
; CHECK-NEXT:    [[X:%.*]] = and i12 [[X_IN:%.*]], 1023
; CHECK-NEXT:    [[Y:%.*]] = and i12 [[Y_IN:%.*]], 1023
; CHECK-NEXT:    [[TMP1:%.*]] = sub nsw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i12 %x_in, 1023
  %y = and i12 %y_in, 1023
  %xf = uitofp i12 %x to half
  %yf = uitofp i12 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i12_sub_fail_overflow(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i12_sub_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = and i12 [[X_IN:%.*]], 1023
; CHECK-NEXT:    [[Y:%.*]] = and i12 [[Y_IN:%.*]], 2047
; CHECK-NEXT:    [[TMP1:%.*]] = sub nsw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i12 %x_in, 1023
  %y = and i12 %y_in, 2047
  %xf = uitofp i12 %x to half
  %yf = uitofp i12 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}


define half @test_si_si_i12_sub(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i12_sub(
; CHECK-NEXT:    [[X:%.*]] = and i12 [[X_IN:%.*]], 1023
; CHECK-NEXT:    [[Y:%.*]] = or i12 [[Y_IN:%.*]], -1024
; CHECK-NEXT:    [[TMP1:%.*]] = sub nsw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i12 %x_in, 1023
  %y = or i12 %y_in, -1024
  %xf = sitofp i12 %x to half
  %yf = sitofp i12 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_si_si_i12_sub_fail_overflow(i12 noundef %x, i12 noundef %y) {
; CHECK-LABEL: @test_si_si_i12_sub_fail_overflow(
; CHECK-NEXT:    [[XF:%.*]] = sitofp i12 [[X:%.*]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i12 [[Y:%.*]] to half
; CHECK-NEXT:    [[R:%.*]] = fsub half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %xf = sitofp i12 %x to half
  %yf = sitofp i12 %y to half
  %r = fsub half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i12_mul(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i12_mul(
; CHECK-NEXT:    [[X:%.*]] = and i12 [[X_IN:%.*]], 31
; CHECK-NEXT:    [[Y:%.*]] = and i12 [[Y_IN:%.*]], 63
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw nsw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i12 %x_in, 31
  %y = and i12 %y_in, 63
  %xf = uitofp i12 %x to half
  %yf = uitofp i12 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i12_mul_fail_overflow(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_ui_ui_i12_mul_fail_overflow(
; CHECK-NEXT:    [[XX:%.*]] = and i12 [[X_IN:%.*]], 31
; CHECK-NEXT:    [[X:%.*]] = add nuw nsw i12 [[XX]], 1
; CHECK-NEXT:    [[Y:%.*]] = and i12 [[Y_IN:%.*]], 63
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i12 %x_in, 31
  %x = add i12 %xx, 1
  %y = and i12 %y_in, 63
  %xf = uitofp i12 %x to half
  %yf = uitofp i12 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_ui_ui_i12_mul_C(i12 noundef %x_in) {
; CHECK-LABEL: @test_ui_ui_i12_mul_C(
; CHECK-NEXT:    [[X:%.*]] = shl i12 [[X_IN:%.*]], 6
; CHECK-NEXT:    [[TMP1:%.*]] = and i12 [[X]], 1984
; CHECK-NEXT:    [[R:%.*]] = uitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i12 %x_in, 31
  %xf = uitofp i12 %x to half
  %r = fmul half %xf, 64.0
  ret half %r
}

define half @test_si_si_i12_mul(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i12_mul(
; CHECK-NEXT:    [[XX:%.*]] = and i12 [[X_IN:%.*]], 30
; CHECK-NEXT:    [[X:%.*]] = or disjoint i12 [[XX]], 1
; CHECK-NEXT:    [[Y:%.*]] = or i12 [[Y_IN:%.*]], -64
; CHECK-NEXT:    [[TMP1:%.*]] = mul nsw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = sitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i12 %x_in, 30
  %x = add nsw nuw i12 %xx, 1
  %y = or i12 %y_in, -64
  %xf = sitofp i12 %x to half
  %yf = sitofp i12 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_si_si_i12_mul_fail_overflow(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i12_mul_fail_overflow(
; CHECK-NEXT:    [[XX:%.*]] = and i12 [[X_IN:%.*]], 30
; CHECK-NEXT:    [[X:%.*]] = or disjoint i12 [[XX]], 1
; CHECK-NEXT:    [[Y:%.*]] = or i12 [[Y_IN:%.*]], -128
; CHECK-NEXT:    [[XF:%.*]] = uitofp i12 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i12 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i12 %x_in, 30
  %x = add nsw nuw i12 %xx, 1
  %y = or i12 %y_in, -128
  %xf = sitofp i12 %x to half
  %yf = sitofp i12 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_si_si_i12_mul_fail_maybe_non_zero(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_si_si_i12_mul_fail_maybe_non_zero(
; CHECK-NEXT:    [[X:%.*]] = and i12 [[X_IN:%.*]], 30
; CHECK-NEXT:    [[Y:%.*]] = or i12 [[Y_IN:%.*]], -128
; CHECK-NEXT:    [[XF:%.*]] = uitofp i12 [[X]] to half
; CHECK-NEXT:    [[YF:%.*]] = sitofp i12 [[Y]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], [[YF]]
; CHECK-NEXT:    ret half [[R]]
;
  %x = and i12 %x_in, 30
  %y = or i12 %y_in, -128
  %xf = sitofp i12 %x to half
  %yf = sitofp i12 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define half @test_si_si_i12_mul_C(i12 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i12_mul_C(
; CHECK-NEXT:    [[X:%.*]] = or i12 [[X_IN:%.*]], -64
; CHECK-NEXT:    [[TMP1:%.*]] = mul nsw i12 [[X]], -16
; CHECK-NEXT:    [[R:%.*]] = uitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i12 %x_in, -64
  %xf = sitofp i12 %x to half
  %r = fmul half %xf, -16.0
  ret half %r
}

define half @test_si_si_i12_mul_C_fail_overflow(i12 noundef %x_in) {
; CHECK-LABEL: @test_si_si_i12_mul_C_fail_overflow(
; CHECK-NEXT:    [[X:%.*]] = or i12 [[X_IN:%.*]], -64
; CHECK-NEXT:    [[XF:%.*]] = sitofp i12 [[X]] to half
; CHECK-NEXT:    [[R:%.*]] = fmul half [[XF]], 0xHD400
; CHECK-NEXT:    ret half [[R]]
;
  %x = or i12 %x_in, -64
  %xf = sitofp i12 %x to half
  %r = fmul half %xf, -64.0
  ret half %r
}

define half @test_ui_si_i12_mul_nsw(i12 noundef %x_in, i12 noundef %y_in) {
; CHECK-LABEL: @test_ui_si_i12_mul_nsw(
; CHECK-NEXT:    [[XX:%.*]] = and i12 [[X_IN:%.*]], 31
; CHECK-NEXT:    [[X:%.*]] = add nuw nsw i12 [[XX]], 1
; CHECK-NEXT:    [[YY:%.*]] = and i12 [[Y_IN:%.*]], 30
; CHECK-NEXT:    [[Y:%.*]] = or disjoint i12 [[YY]], 1
; CHECK-NEXT:    [[TMP1:%.*]] = mul nuw nsw i12 [[X]], [[Y]]
; CHECK-NEXT:    [[R:%.*]] = uitofp i12 [[TMP1]] to half
; CHECK-NEXT:    ret half [[R]]
;
  %xx = and i12 %x_in, 31
  %x = add i12 %xx, 1
  %yy = and i12 %y_in, 30
  %y = add i12 %yy, 1
  %xf = uitofp i12 %x to half
  %yf = sitofp i12 %y to half
  %r = fmul half %xf, %yf
  ret half %r
}

define float @test_ui_add_with_signed_constant(i32 %shr.i) {
; CHECK-LABEL: @test_ui_add_with_signed_constant(
; CHECK-NEXT:    [[AND_I:%.*]] = and i32 [[SHR_I:%.*]], 32767
; CHECK-NEXT:    [[TMP1:%.*]] = add nsw i32 [[AND_I]], -16383
; CHECK-NEXT:    [[ADD:%.*]] = sitofp i32 [[TMP1]] to float
; CHECK-NEXT:    ret float [[ADD]]
;
  %and.i = and i32 %shr.i, 32767
  %sub = uitofp i32 %and.i to float
  %add = fadd float %sub, -16383.0
  ret float %add
}


;; Reduced form of bug noticed due to #82555
define float @missed_nonzero_check_on_constant_for_si_fmul(i1 %c, i1 %.b, ptr %g_2345) {
; CHECK-LABEL: @missed_nonzero_check_on_constant_for_si_fmul(
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[C:%.*]], i32 65529, i32 53264
; CHECK-NEXT:    [[CONV_I:%.*]] = trunc i32 [[SEL]] to i16
; CHECK-NEXT:    [[CONV1_I:%.*]] = sitofp i16 [[CONV_I]] to float
; CHECK-NEXT:    [[MUL3_I_I:%.*]] = fmul float [[CONV1_I]], 0.000000e+00
; CHECK-NEXT:    store i32 [[SEL]], ptr [[G_2345:%.*]], align 4
; CHECK-NEXT:    ret float [[MUL3_I_I]]
;
  %sel = select i1 %c, i32 65529, i32 53264
  %conv.i = trunc i32 %sel to i16
  %conv1.i = sitofp i16 %conv.i to float
  %mul3.i.i = fmul float %conv1.i, 0.000000e+00
  store i32 %sel, ptr %g_2345, align 4
  ret float %mul3.i.i
}

define <2 x float> @missed_nonzero_check_on_constant_for_si_fmul_vec(i1 %c, i1 %.b, ptr %g_2345) {
; CHECK-LABEL: @missed_nonzero_check_on_constant_for_si_fmul_vec(
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[C:%.*]], i32 65529, i32 53264
; CHECK-NEXT:    [[CONV_I_S:%.*]] = trunc i32 [[SEL]] to i16
; CHECK-NEXT:    [[CONV_I_V:%.*]] = insertelement <2 x i16> poison, i16 [[CONV_I_S]], i64 0
; CHECK-NEXT:    [[CONV_I:%.*]] = shufflevector <2 x i16> [[CONV_I_V]], <2 x i16> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[CONV1_I:%.*]] = sitofp <2 x i16> [[CONV_I]] to <2 x float>
; CHECK-NEXT:    [[MUL3_I_I:%.*]] = fmul <2 x float> [[CONV1_I]], zeroinitializer
; CHECK-NEXT:    store i32 [[SEL]], ptr [[G_2345:%.*]], align 4
; CHECK-NEXT:    ret <2 x float> [[MUL3_I_I]]
;
  %sel = select i1 %c, i32 65529, i32 53264
  %conv.i.s = trunc i32 %sel to i16
  %conv.i.v = insertelement <2 x i16> poison, i16 %conv.i.s, i64 0
  %conv.i = insertelement <2 x i16> %conv.i.v, i16 %conv.i.s, i64 1
  %conv1.i = sitofp <2 x i16> %conv.i to <2 x float>
  %mul3.i.i = fmul <2 x float> %conv1.i, zeroinitializer
  store i32 %sel, ptr %g_2345, align 4
  ret <2 x float> %mul3.i.i
}

define float @negzero_check_on_constant_for_si_fmul(i1 %c, i1 %.b, ptr %g_2345) {
; CHECK-LABEL: @negzero_check_on_constant_for_si_fmul(
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[C:%.*]], i32 65529, i32 53264
; CHECK-NEXT:    [[CONV_I:%.*]] = trunc i32 [[SEL]] to i16
; CHECK-NEXT:    [[CONV1_I:%.*]] = sitofp i16 [[CONV_I]] to float
; CHECK-NEXT:    [[MUL3_I_I:%.*]] = fmul float [[CONV1_I]], -0.000000e+00
; CHECK-NEXT:    store i32 [[SEL]], ptr [[G_2345:%.*]], align 4
; CHECK-NEXT:    ret float [[MUL3_I_I]]
;
  %sel = select i1 %c, i32 65529, i32 53264
  %conv.i = trunc i32 %sel to i16
  %conv1.i = sitofp i16 %conv.i to float
  %mul3.i.i = fmul float %conv1.i, -0.000000e+00
  store i32 %sel, ptr %g_2345, align 4
  ret float %mul3.i.i
}

define <2 x float> @nonzero_check_on_constant_for_si_fmul_vec_w_undef(i1 %c, i1 %.b, ptr %g_2345) {
; CHECK-LABEL: @nonzero_check_on_constant_for_si_fmul_vec_w_undef(
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[C:%.*]], i32 65529, i32 53264
; CHECK-NEXT:    [[CONV_I_S:%.*]] = trunc i32 [[SEL]] to i16
; CHECK-NEXT:    [[CONV_I_V:%.*]] = insertelement <2 x i16> poison, i16 [[CONV_I_S]], i64 0
; CHECK-NEXT:    [[CONV_I:%.*]] = shufflevector <2 x i16> [[CONV_I_V]], <2 x i16> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[CONV1_I:%.*]] = sitofp <2 x i16> [[CONV_I]] to <2 x float>
; CHECK-NEXT:    [[MUL3_I_I:%.*]] = fmul <2 x float> [[CONV1_I]], <float undef, float 0.000000e+00>
; CHECK-NEXT:    store i32 [[SEL]], ptr [[G_2345:%.*]], align 4
; CHECK-NEXT:    ret <2 x float> [[MUL3_I_I]]
;
  %sel = select i1 %c, i32 65529, i32 53264
  %conv.i.s = trunc i32 %sel to i16
  %conv.i.v = insertelement <2 x i16> poison, i16 %conv.i.s, i64 0
  %conv.i = insertelement <2 x i16> %conv.i.v, i16 %conv.i.s, i64 1
  %conv1.i = sitofp <2 x i16> %conv.i to <2 x float>
  %mul3.i.i = fmul <2 x float> %conv1.i, <float undef, float 0.000000e+00>
  store i32 %sel, ptr %g_2345, align 4
  ret <2 x float> %mul3.i.i
}

define <2 x float> @nonzero_check_on_constant_for_si_fmul_nz_vec_w_undef(i1 %c, i1 %.b, ptr %g_2345) {
; CHECK-LABEL: @nonzero_check_on_constant_for_si_fmul_nz_vec_w_undef(
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[C:%.*]], i32 65529, i32 53264
; CHECK-NEXT:    [[CONV_I_S:%.*]] = trunc i32 [[SEL]] to i16
; CHECK-NEXT:    [[CONV_I_V:%.*]] = insertelement <2 x i16> poison, i16 [[CONV_I_S]], i64 0
; CHECK-NEXT:    [[CONV_I:%.*]] = shufflevector <2 x i16> [[CONV_I_V]], <2 x i16> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[CONV1_I:%.*]] = sitofp <2 x i16> [[CONV_I]] to <2 x float>
; CHECK-NEXT:    [[MUL3_I_I:%.*]] = fmul <2 x float> [[CONV1_I]], <float undef, float 1.000000e+00>
; CHECK-NEXT:    store i32 [[SEL]], ptr [[G_2345:%.*]], align 4
; CHECK-NEXT:    ret <2 x float> [[MUL3_I_I]]
;
  %sel = select i1 %c, i32 65529, i32 53264
  %conv.i.s = trunc i32 %sel to i16
  %conv.i.v = insertelement <2 x i16> poison, i16 %conv.i.s, i64 0
  %conv.i = insertelement <2 x i16> %conv.i.v, i16 %conv.i.s, i64 1
  %conv1.i = sitofp <2 x i16> %conv.i to <2 x float>
  %mul3.i.i = fmul <2 x float> %conv1.i, <float undef, float 1.000000e+00>
  store i32 %sel, ptr %g_2345, align 4
  ret <2 x float> %mul3.i.i
}

define <2 x float> @nonzero_check_on_constant_for_si_fmul_negz_vec_w_undef(i1 %c, i1 %.b, ptr %g_2345) {
; CHECK-LABEL: @nonzero_check_on_constant_for_si_fmul_negz_vec_w_undef(
; CHECK-NEXT:    [[SEL:%.*]] = select i1 [[C:%.*]], i32 65529, i32 53264
; CHECK-NEXT:    [[CONV_I_S:%.*]] = trunc i32 [[SEL]] to i16
; CHECK-NEXT:    [[CONV_I_V:%.*]] = insertelement <2 x i16> poison, i16 [[CONV_I_S]], i64 0
; CHECK-NEXT:    [[CONV_I:%.*]] = shufflevector <2 x i16> [[CONV_I_V]], <2 x i16> poison, <2 x i32> zeroinitializer
; CHECK-NEXT:    [[CONV1_I:%.*]] = sitofp <2 x i16> [[CONV_I]] to <2 x float>
; CHECK-NEXT:    [[MUL3_I_I:%.*]] = fmul <2 x float> [[CONV1_I]], <float undef, float -0.000000e+00>
; CHECK-NEXT:    store i32 [[SEL]], ptr [[G_2345:%.*]], align 4
; CHECK-NEXT:    ret <2 x float> [[MUL3_I_I]]
;
  %sel = select i1 %c, i32 65529, i32 53264
  %conv.i.s = trunc i32 %sel to i16
  %conv.i.v = insertelement <2 x i16> poison, i16 %conv.i.s, i64 0
  %conv.i = insertelement <2 x i16> %conv.i.v, i16 %conv.i.s, i64 1
  %conv1.i = sitofp <2 x i16> %conv.i to <2 x float>
  %mul3.i.i = fmul <2 x float> %conv1.i, <float undef, float -0.000000e+00>
  store i32 %sel, ptr %g_2345, align 4
  ret <2 x float> %mul3.i.i
}
