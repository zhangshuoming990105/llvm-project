; NOTE: Assertions have been autogenerated by utils/update_test_checks.py UTC_ARGS: --version 3
; RUN: opt -S --passes=slp-vectorizer -mtriple=x86_64-unknown-linux-gnu -mcpu=alderlake < %s| FileCheck %s

define void @test() {
; CHECK-LABEL: define void @test(
; CHECK-SAME: ) #[[ATTR0:[0-9]+]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX22:%.*]] = getelementptr i32, ptr null, i64 60
; CHECK-NEXT:    [[TMP0:%.*]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> getelementptr (i32, <4 x ptr> zeroinitializer, <4 x i64> <i64 1, i64 33, i64 7, i64 0>), i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> poison)
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr [[ARRAYIDX22]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT:    [[TMP3:%.*]] = mul <4 x i32> [[TMP2]], [[TMP0]]
; CHECK-NEXT:    [[TMP4:%.*]] = sext <4 x i32> [[TMP3]] to <4 x i64>
; CHECK-NEXT:    [[TMP5:%.*]] = ashr <4 x i64> [[TMP4]], zeroinitializer
; CHECK-NEXT:    [[TMP6:%.*]] = trunc <4 x i64> [[TMP5]] to <4 x i32>
; CHECK-NEXT:    store <4 x i32> [[TMP6]], ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 0), align 16
; CHECK-NEXT:    ret void
;
entry:
  %arrayidx1 = getelementptr i32, ptr null, i64 1
  %0 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr i32, ptr null, i64 63
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul i32 %1, %0
  %conv = sext i32 %mul to i64
  %shr = ashr i64 %conv, 0
  %conv3 = trunc i64 %shr to i32
  store i32 %conv3, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 0), align 16
  %arrayidx5 = getelementptr i32, ptr null, i64 33
  %2 = load i32, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr i32, ptr null, i64 62
  %3 = load i32, ptr %arrayidx6, align 4
  %mul7 = mul i32 %3, %2
  %conv8 = sext i32 %mul7 to i64
  %shr10 = ashr i64 %conv8, 0
  %conv11 = trunc i64 %shr10 to i32
  store i32 %conv11, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 1), align 4
  %arrayidx13 = getelementptr i32, ptr null, i64 7
  %4 = load i32, ptr %arrayidx13, align 4
  %arrayidx14 = getelementptr i32, ptr null, i64 61
  %5 = load i32, ptr %arrayidx14, align 4
  %mul15 = mul i32 %5, %4
  %conv16 = sext i32 %mul15 to i64
  %shr18 = ashr i64 %conv16, 0
  %conv19 = trunc i64 %shr18 to i32
  store i32 %conv19, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 2), align 8
  %6 = load i32, ptr null, align 4
  %arrayidx22 = getelementptr i32, ptr null, i64 60
  %7 = load i32, ptr %arrayidx22, align 4
  %mul23 = mul i32 %7, %6
  %conv24 = sext i32 %mul23 to i64
  %shr26 = ashr i64 %conv24, 0
  %conv27 = trunc i64 %shr26 to i32
  store i32 %conv27, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 3), align 4
  ret void
}

define void @test1() {
; CHECK-LABEL: define void @test1(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX22:%.*]] = getelementptr i32, ptr null, i64 60
; CHECK-NEXT:    [[TMP0:%.*]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> getelementptr (i32, <4 x ptr> zeroinitializer, <4 x i64> <i64 1, i64 33, i64 7, i64 0>), i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> poison)
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr [[ARRAYIDX22]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT:    [[TMP3:%.*]] = mul <4 x i32> [[TMP2]], [[TMP0]]
; CHECK-NEXT:    [[TMP4:%.*]] = sext <4 x i32> [[TMP3]] to <4 x i64>
; CHECK-NEXT:    [[TMP5:%.*]] = lshr <4 x i64> [[TMP4]], zeroinitializer
; CHECK-NEXT:    [[TMP6:%.*]] = trunc <4 x i64> [[TMP5]] to <4 x i32>
; CHECK-NEXT:    store <4 x i32> [[TMP6]], ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 0), align 16
; CHECK-NEXT:    ret void
;
entry:
  %arrayidx1 = getelementptr i32, ptr null, i64 1
  %0 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr i32, ptr null, i64 63
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul i32 %1, %0
  %conv = sext i32 %mul to i64
  %shr = lshr i64 %conv, 0
  %conv3 = trunc i64 %shr to i32
  store i32 %conv3, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 0), align 16
  %arrayidx5 = getelementptr i32, ptr null, i64 33
  %2 = load i32, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr i32, ptr null, i64 62
  %3 = load i32, ptr %arrayidx6, align 4
  %mul7 = mul i32 %3, %2
  %conv8 = sext i32 %mul7 to i64
  %shr10 = lshr i64 %conv8, 0
  %conv11 = trunc i64 %shr10 to i32
  store i32 %conv11, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 1), align 4
  %arrayidx13 = getelementptr i32, ptr null, i64 7
  %4 = load i32, ptr %arrayidx13, align 4
  %arrayidx14 = getelementptr i32, ptr null, i64 61
  %5 = load i32, ptr %arrayidx14, align 4
  %mul15 = mul i32 %5, %4
  %conv16 = sext i32 %mul15 to i64
  %shr18 = lshr i64 %conv16, 0
  %conv19 = trunc i64 %shr18 to i32
  store i32 %conv19, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 2), align 8
  %6 = load i32, ptr null, align 4
  %arrayidx22 = getelementptr i32, ptr null, i64 60
  %7 = load i32, ptr %arrayidx22, align 4
  %mul23 = mul i32 %7, %6
  %conv24 = sext i32 %mul23 to i64
  %shr26 = lshr i64 %conv24, 0
  %conv27 = trunc i64 %shr26 to i32
  store i32 %conv27, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 3), align 4
  ret void
}

define void @test_div() {
; CHECK-LABEL: define void @test_div(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX22:%.*]] = getelementptr i32, ptr null, i64 60
; CHECK-NEXT:    [[TMP0:%.*]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> getelementptr (i32, <4 x ptr> zeroinitializer, <4 x i64> <i64 1, i64 33, i64 7, i64 0>), i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> poison)
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr [[ARRAYIDX22]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT:    [[TMP3:%.*]] = mul <4 x i32> [[TMP2]], [[TMP0]]
; CHECK-NEXT:    [[TMP4:%.*]] = zext <4 x i32> [[TMP3]] to <4 x i64>
; CHECK-NEXT:    [[TMP5:%.*]] = udiv <4 x i64> [[TMP4]], <i64 1, i64 2, i64 1, i64 2>
; CHECK-NEXT:    [[TMP6:%.*]] = trunc <4 x i64> [[TMP5]] to <4 x i32>
; CHECK-NEXT:    store <4 x i32> [[TMP6]], ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 0), align 16
; CHECK-NEXT:    ret void
;
entry:
  %arrayidx1 = getelementptr i32, ptr null, i64 1
  %0 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr i32, ptr null, i64 63
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul i32 %1, %0
  %conv = zext i32 %mul to i64
  %shr = udiv i64 %conv, 1
  %conv3 = trunc i64 %shr to i32
  store i32 %conv3, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 0), align 16
  %arrayidx5 = getelementptr i32, ptr null, i64 33
  %2 = load i32, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr i32, ptr null, i64 62
  %3 = load i32, ptr %arrayidx6, align 4
  %mul7 = mul i32 %3, %2
  %conv8 = zext i32 %mul7 to i64
  %shr10 = udiv i64 %conv8, 2
  %conv11 = trunc i64 %shr10 to i32
  store i32 %conv11, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 1), align 4
  %arrayidx13 = getelementptr i32, ptr null, i64 7
  %4 = load i32, ptr %arrayidx13, align 4
  %arrayidx14 = getelementptr i32, ptr null, i64 61
  %5 = load i32, ptr %arrayidx14, align 4
  %mul15 = mul i32 %5, %4
  %conv16 = zext i32 %mul15 to i64
  %shr18 = udiv i64 %conv16, 1
  %conv19 = trunc i64 %shr18 to i32
  store i32 %conv19, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 2), align 8
  %6 = load i32, ptr null, align 4
  %arrayidx22 = getelementptr i32, ptr null, i64 60
  %7 = load i32, ptr %arrayidx22, align 4
  %mul23 = mul i32 %7, %6
  %conv24 = zext i32 %mul23 to i64
  %shr26 = udiv i64 %conv24, 2
  %conv27 = trunc i64 %shr26 to i32
  store i32 %conv27, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 3), align 4
  ret void
}

define void @test_rem() {
; CHECK-LABEL: define void @test_rem(
; CHECK-SAME: ) #[[ATTR0]] {
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ARRAYIDX22:%.*]] = getelementptr i32, ptr null, i64 60
; CHECK-NEXT:    [[TMP0:%.*]] = call <4 x i32> @llvm.masked.gather.v4i32.v4p0(<4 x ptr> getelementptr (i32, <4 x ptr> zeroinitializer, <4 x i64> <i64 1, i64 33, i64 7, i64 0>), i32 4, <4 x i1> <i1 true, i1 true, i1 true, i1 true>, <4 x i32> poison)
; CHECK-NEXT:    [[TMP1:%.*]] = load <4 x i32>, ptr [[ARRAYIDX22]], align 4
; CHECK-NEXT:    [[TMP2:%.*]] = shufflevector <4 x i32> [[TMP1]], <4 x i32> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>
; CHECK-NEXT:    [[TMP3:%.*]] = mul <4 x i32> [[TMP2]], [[TMP0]]
; CHECK-NEXT:    [[TMP4:%.*]] = zext <4 x i32> [[TMP3]] to <4 x i64>
; CHECK-NEXT:    [[TMP5:%.*]] = urem <4 x i64> [[TMP4]], <i64 1, i64 2, i64 1, i64 1>
; CHECK-NEXT:    [[TMP6:%.*]] = trunc <4 x i64> [[TMP5]] to <4 x i32>
; CHECK-NEXT:    store <4 x i32> [[TMP6]], ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 0), align 16
; CHECK-NEXT:    ret void
;
entry:
  %arrayidx1 = getelementptr i32, ptr null, i64 1
  %0 = load i32, ptr %arrayidx1, align 4
  %arrayidx2 = getelementptr i32, ptr null, i64 63
  %1 = load i32, ptr %arrayidx2, align 4
  %mul = mul i32 %1, %0
  %conv = zext i32 %mul to i64
  %shr = urem i64 %conv, 1
  %conv3 = trunc i64 %shr to i32
  store i32 %conv3, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 0), align 16
  %arrayidx5 = getelementptr i32, ptr null, i64 33
  %2 = load i32, ptr %arrayidx5, align 4
  %arrayidx6 = getelementptr i32, ptr null, i64 62
  %3 = load i32, ptr %arrayidx6, align 4
  %mul7 = mul i32 %3, %2
  %conv8 = zext i32 %mul7 to i64
  %shr10 = urem i64 %conv8, 2
  %conv11 = trunc i64 %shr10 to i32
  store i32 %conv11, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 1), align 4
  %arrayidx13 = getelementptr i32, ptr null, i64 7
  %4 = load i32, ptr %arrayidx13, align 4
  %arrayidx14 = getelementptr i32, ptr null, i64 61
  %5 = load i32, ptr %arrayidx14, align 4
  %mul15 = mul i32 %5, %4
  %conv16 = zext i32 %mul15 to i64
  %shr18 = urem i64 %conv16, 1
  %conv19 = trunc i64 %shr18 to i32
  store i32 %conv19, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 2), align 8
  %6 = load i32, ptr null, align 4
  %arrayidx22 = getelementptr i32, ptr null, i64 60
  %7 = load i32, ptr %arrayidx22, align 4
  %mul23 = mul i32 %7, %6
  %conv24 = zext i32 %mul23 to i64
  %shr26 = urem i64 %conv24, 1
  %conv27 = trunc i64 %shr26 to i32
  store i32 %conv27, ptr getelementptr inbounds ([4 x i32], ptr null, i64 8, i64 3), align 4
  ret void
}
