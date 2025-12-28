#pragma once

#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <thread>
#include "panna/channel.hpp"
#include "panna/logging.hpp"

namespace panna {
    TEST_CASE( "channels one-to-one" ) {
        Channel<uint32_t> chan(16);

        std::vector<uint32_t> tosend(100);
        std::iota(tosend.begin(), tosend.end(), 0);
        std::vector<uint32_t> received;

        std::thread consumer( [&] {
            for ( std::optional<int> msg = chan.receive(); msg.has_value(); msg = chan.receive() ) {
                received.push_back(*msg);
            }
        } );

        std::thread producer( [&] {
            for ( auto i : tosend ) {
                chan.send( std::move( i ) );
            }
            chan.close();
        } );

        producer.join();
        consumer.join();

        REQUIRE(tosend == received);
    }

    TEST_CASE( "channels many-to-one" ) {
        using namespace std::chrono_literals;

        Channel<uint32_t> chan(16);

        std::vector<uint32_t> received;

        size_t num_producers = 32;
        size_t num_messages = 50;
        std::thread consumer( [&] {
            for ( std::optional<uint32_t> msg = chan.receive(); msg.has_value(); msg = chan.receive() ) {
                received.push_back(*msg);
                if (received.size() >= num_producers*num_messages) {
                    break;
                }
            }
        } );

        std::vector<uint32_t> expected;
        std::vector<std::thread> producers;
        for ( size_t mtid = 0; mtid < num_producers; mtid++ ) {
            size_t tid = mtid;
            std::thread producer( [&chan, tid, num_messages] {
                for ( size_t i = 0; i < num_messages; i++ ) {
                    chan.send( i + (tid*100) );
                }
            } );
            producers.push_back( std::move( producer ) );
            for ( size_t i = 0; i < num_messages; i++ ) {
                expected.push_back( i + (tid*100) );
            }
        }

        for ( auto&& p : producers ) {
            p.join();
        }
        consumer.join();

        std::sort( received.begin(), received.end() );
        std::sort( expected.begin(), expected.end() );

        REQUIRE( expected.size() == received.size() );
        for ( size_t i = 0; i < expected.size(); i++ ) {
            REQUIRE( expected[i] == received[i] );
        }
    }
} // namespace panna
