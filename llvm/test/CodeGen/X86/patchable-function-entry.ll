; RUN: llc -mtriple=i386 %s -o - | FileCheck --check-prefixes=CHECK,NOFSECT,32 %s
; RUN: llc -mtriple=x86_64 %s -o - | FileCheck --check-prefixes=CHECK,NOFSECT,64 %s
; RUN: llc -mtriple=x86_64 -function-sections %s -o - | FileCheck --check-prefixes=CHECK,FSECT,64 %s

define void @f0() "patchable-function-entry"="0" {
; CHECK-LABEL: f0:
; CHECK-NEXT: .Lfunc_begin0:
; CHECK-NOT:   nop
; CHECK:       ret
; CHECK-NOT:   .section __patchable_function_entries
  ret void
}

define void @f1() "patchable-function-entry"="1" {
; CHECK-LABEL: f1:
; CHECK-NEXT: .Lfunc_begin1:
; CHECK:       nop
; CHECK-NEXT:  ret
; CHECK:       .section __patchable_function_entries,"awo",@progbits,f1,unique,0
; 32:          .p2align 2
; 32-NEXT:     .long .Lfunc_begin1
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin1
  ret void
}

define void @f2() "patchable-function-entry"="2" {
; CHECK-LABEL: f2:
; 32-COUNT-2:  nop
; 64:          xchgw %ax, %ax
; CHECK-NEXT:  ret
; NOFSECT:     .section __patchable_function_entries,"awo",@progbits,f1,unique,0
; FSECT:       .section __patchable_function_entries,"awo",@progbits,f2,unique,1
; 32:          .p2align 2
; 32-NEXT:     .long .Lfunc_begin2
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin2
  ret void
}

$f3 = comdat any
define void @f3() "patchable-function-entry"="3" comdat {
; CHECK-LABEL: f3:
; 32-COUNT-3:  nop
; 64:          nopl (%rax)
; CHECK:       ret
; NOFSECT:     .section __patchable_function_entries,"aGwo",@progbits,f3,comdat,f3,unique,1
; FSECT:       .section __patchable_function_entries,"aGwo",@progbits,f3,comdat,f3,unique,2
; 32:          .p2align 2
; 32-NEXT:     .long .Lfunc_begin3
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin3
  ret void
}

$f5 = comdat any
define void @f5() "patchable-function-entry"="5" comdat {
; CHECK-LABEL: f5:
; 32-COUNT-5:  nop
; 64:          nopl 8(%rax,%rax)
; CHECK-NEXT:  ret
; NOFSECT      .section __patchable_function_entries,"aGwo",@progbits,f5,comdat,f5,unique,2
; FSECT:       .section __patchable_function_entries,"aGwo",@progbits,f5,comdat,f5,unique,3
; 32:          .p2align 2
; 32-NEXT:     .long .Lfunc_begin4
; 64:          .p2align 3
; 64-NEXT:     .quad .Lfunc_begin4
  ret void
}
