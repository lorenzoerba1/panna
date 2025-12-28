#pragma once

#include <deque>
#include <mutex>
#include <optional>
#include <semaphore>

namespace panna {
    /// A communication channel between threads. It has a fixed size, and readers
    /// and writers block waiting for either space or new data elements, if either
    /// is lacking.
    template <typename T>
    struct Channel {
    private:
        /// Does the channel accept new messages?
        bool closed;
        /// How many messages can be in flight at any one time?
        size_t capacity;
        /// The actual messages
        std::deque<T> messages;
        /// Synchronization primitive: allows waiting if there is no place to
        /// write a new message
        std::counting_semaphore<> available_slots;
        /// Synchronization primitive: allows waiting if there are no
        /// messages to pop from the queue
        std::counting_semaphore<> available_messages;
        /// Prevent concurrent modifications to the queue
        std::mutex mutex;

    public:
        explicit Channel( size_t capacity ):
            closed( false ),
            capacity( capacity ),
            messages( 0 ),
            available_slots( capacity ),
            available_messages( 0 ),
            mutex() {
        }

        /// Send an items (moving it) and returns `true` if the message has been sent
        bool send( T&& message ) {
            if (closed) {
                return false;
            }
            // mark a slot as occupied, or wait for space to be available
            available_slots.acquire();
            // acquire lock
            std::lock_guard<std::mutex> lock(mutex);
            if (closed) {
                available_slots.release();
                return false;
            }
            messages.push_back(std::move(message));
            // signal that there are messages available
            available_messages.release();
            return true;
        }

        std::optional<T> receive(){
            // potentially wait for new messages to be available
            available_messages.acquire();
            std::lock_guard<std::mutex> lock(mutex);
            if (messages.empty()) {
                // the queue has been closed and emptied
                return std::nullopt;
            }
            T msg = std::move(messages.front());
            messages.pop_front();
            // mark a slot as available
            available_slots.release();
            return msg;
        }

        void close() {
            std::lock_guard<std::mutex> lock(mutex);
            closed = true;
            // unblock all waiting threads
            available_slots.release(capacity);
            available_messages.release(capacity);
        }
    };

} // namespace panna
