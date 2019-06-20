#ifndef FUSION_MAPPING_HASH_ENTRY_H
#define FUSION_MAPPING_HASH_ENTRY_H

#include <fusion/macros.h>
#include <fusion/math/matrix.h>
#include <fusion/math/vector.h>

namespace fusion
{

struct FUSION_EXPORT HashEntry
{
    FUSION_HOST_AND_DEVICE inline HashEntry();
    FUSION_HOST_AND_DEVICE inline HashEntry(Vector3i pos_, int next, int offset_);
    FUSION_HOST_AND_DEVICE inline HashEntry(const HashEntry &);
    FUSION_HOST_AND_DEVICE inline HashEntry &operator=(const HashEntry &);
    FUSION_HOST_AND_DEVICE inline bool operator==(const Vector3i &) const;
    FUSION_HOST_AND_DEVICE inline bool operator==(const HashEntry &) const;

    int ptr_;
    int offset_;
    Vector3i pos_;
};

FUSION_HOST_AND_DEVICE inline HashEntry::HashEntry()
    : ptr_(-1), offset_(-1)
{
}

FUSION_HOST_AND_DEVICE inline HashEntry::HashEntry(Vector3i pos_, int ptr_, int offset_)
    : pos_(pos_), ptr_(ptr_), offset_(offset_)
{
}

FUSION_HOST_AND_DEVICE inline HashEntry::HashEntry(const HashEntry &other)
    : pos_(other.pos_), ptr_(other.ptr_), offset_(other.offset_)
{
}

FUSION_HOST_AND_DEVICE inline HashEntry &HashEntry::operator=(const HashEntry &other)
{
    pos_ = other.pos_;
    ptr_ = other.ptr_;
    offset_ = other.offset_;
    return *this;
}

FUSION_HOST_AND_DEVICE inline bool HashEntry::operator==(const Vector3i &pos_) const
{
    return this->pos_ == pos_;
}

FUSION_HOST_AND_DEVICE inline bool HashEntry::operator==(const HashEntry &other) const
{
    return other.pos_ == pos_;
}

} // namespace fusion

#endif