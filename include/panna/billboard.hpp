#pragma once

#include <atomic>
#include <memory>

/// A data structure that allows a writer to publish a value to be read
/// by many readers. Updates happen atomically.
template <typename T>
class Billboard {
private:
    std::atomic<std::shared_ptr<T>> ptr;

public:
    Billboard(): ptr( std::make_shared<T>() ) {
    }

    /// Replace the published object, claiming ownership of the new one
    void update( T&& value ) {
        auto p = std::make_shared<T>( std::move( value ) );
        ptr.store( p, std::memory_order_release );
    }

    /// Get a snapshot of the published object for const reading
    std::shared_ptr<const T> read() const {
        return ptr.load( std::memory_order_acquire );
    }
};
